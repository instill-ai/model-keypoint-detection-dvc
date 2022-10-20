from typing import List, Tuple

import numpy as np
from triton_python_model.task.keypoint import PostKeypointDetectionModel


class TritonPythonModel(PostKeypointDetectionModel):
    def __init__(self):
        super().__init__(['pred_keypoints', 'pred_scores', 'pred_boxes', 'scale', 'pad'], ['kpoints', 'boxes', 'scores'])

    def post_process_per_image(self, inputs: Tuple[np.ndarray]) -> List[np.ndarray]:
        pred_keypoints = inputs[0] # batching result
        scores = inputs[1] # batching result
        boxes = inputs[2] # batching result
        scales = inputs[3]  # batching result
        pads = inputs[4]  # batching result

        batch_kps = []
        batch_scs = []
        batch_bbs = []
        for image_pred_keypoints, image_boxes, image_scores, image_scales, image_pads in zip(pred_keypoints, boxes, scores, scales, pads): # single image result
            image_kps = []
            image_bbs = []
            image_scs = []
            for i, score in enumerate(image_scores):
                if score < 0.8:
                    continue
                kps = image_pred_keypoints[i]
                obj_kps = []
                for kp in kps:
                    obj_kps.append([(kp[0]-image_pads[0])/image_scales[0], (kp[1]-image_pads[1])/image_scales[1], kp[2]])
                image_kps.append(obj_kps)
                x, y, w, h = image_boxes[i]
                x = int((x-image_pads[0])/image_scales[0])
                y = int((y-image_pads[1])/image_scales[1])
                w = int((w-image_pads[0])/image_scales[0] - x)
                h = int((h-image_pads[1])/image_scales[1] - y)
                image_bbs.append([x, y, w, h])
                image_scs.append(score)
            batch_kps.append(image_kps)
            batch_scs.append(image_scs)
            batch_bbs.append(image_bbs)

        max_objs = max([len(i) for i in batch_bbs])
        for kps, bbs, scs in zip(batch_kps, batch_bbs, batch_scs): # add dummy data to make sure same shape for broadcast result
            for _ in range(max_objs - len(bbs)):
                kps.append([[-1, -1, -1]] * 17)
                bbs.append([-1, -1, -1, -1])
                scs.append(0)
        return [np.array(batch_kps), np.array(batch_bbs), np.array(batch_scs)]
