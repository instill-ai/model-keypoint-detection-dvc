from typing import List, Tuple

import numpy as np
from triton_python_model.task.keypoint import PostKeypointDetectionModel


class TritonPythonModel(PostKeypointDetectionModel):
    def __init__(self):
        super().__init__(['pred_keypoints', 'pred_scores', 'pred_boxes', 'scale'], ['kpoints', 'scores'])

    def post_process_per_image(self, inputs: Tuple[np.ndarray]) -> List[np.ndarray]:
        pred_keypoints = inputs[0]
        scores = inputs[1][0]
        scales = inputs[3]  # inputs[2] is bounding box of person, not use yet
        kps = []
        scs = []
        for i, s in enumerate(scores):
            if s < 0.8:
                continue
            kps.append(pred_keypoints[i])
            scs.append(s)
        kps = [p / scales[0] for p in kps]
        return [np.array(kps), np.array(scs)]
