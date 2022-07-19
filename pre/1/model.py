import io
import numpy as np
import json

from typing import List
from PIL import Image

from triton_python_backend_utils import Tensor, InferenceResponse, \
    get_input_tensor_by_name, InferenceRequest, get_output_config_by_name, triton_string_to_numpy

def scale_up(img, min_size=800):
    w, h = img.size
    if w < h:
        scale = min_size / w
        new_size = (min_size, h*min_size//w)
    else:
        scale = min_size / h
        new_size = (w*min_size//h, min_size)
    img = img.resize(new_size)
    return img, scale

class TritonPythonModel(object):
    def __init__(self):
        self.tf = None
        self.output_names = {
            'image': 'image',
            'scale': 'scale'
        }

    def initialize(self, args):
        model_config = json.loads(args['model_config'])

        output_configs = {k: get_output_config_by_name(
            model_config, name) for k, name in self.output_names.items()}
        for k, cfg in output_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Output {self.output_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for output {self.output_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for output {self.output_names[k]} is not defined in the model config')
            if 'data_type' not in cfg:
                raise ValueError(
                    f'Data type for output {self.output_names[k]} is not defined in the model config')

        self.output_dtypes = {k: triton_string_to_numpy(
            cfg['data_type']) for k, cfg in output_configs.items()}


    def execute(self, inference_requests: List[InferenceRequest]) -> List[InferenceResponse]:
        input_name = 'input'

        responses = []
        for request in inference_requests:
            # This model only process one input per request. We use
            # get_input_tensor_by_name instead of checking
            # len(request.inputs()) to allow for multiple inputs but
            # only process the one we want. Same rationale for the outputs
            batch_in_tensor: Tensor = get_input_tensor_by_name(request, input_name)
            batch_in = batch_in_tensor.as_numpy()  # shape (batch_size, 1)

            if batch_in.dtype.type is not np.object_:
                raise ValueError(f'Input datatype must be np.object_, ' f'got {batch_in.dtype.type}')
            
            batch_out = {k: [] for k, name in self.output_names.items(
            ) if name in request.requested_output_names()}

            for img in batch_in:  # img is shape (1,)
                pil_img = Image.open(io.BytesIO(img.astype(bytes)))
                img, scale = scale_up(pil_img)
                img = np.array(img)
                img = np.transpose(img, (2, 0, 1))
                batch_out['image'].append(np.array(img))
                batch_out['scale'].append([scale])
            # Format outputs to build an InferenceResponse
            # Assumes there is only one output
            output_tensors = [Tensor(self.output_names[k], np.asarray(
                out, dtype=self.output_dtypes[k])) for k, out in batch_out.items()]

            # TODO: should set error field from InferenceResponse constructor
            # to handle errors
            response = InferenceResponse(output_tensors)
            responses.append(response)

        return responses
