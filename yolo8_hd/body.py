
import os
import numpy as np
import cv2
from yolo8_hd.head import getCenter

from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_b"](checkpoint="./yolo8_hd/sam_vit_b_01ec64.pth")
#sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
#sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")

predictor = SamPredictor(sam)

def getMask(image_arr):
    x, y = getCenter(image_arr)
    predictor.set_image(image_arr)

    masks, scores, logits = predictor.predict(
            point_coords=np.array([[x,y]]),
            point_labels=np.array([1]),
            multimask_output=True,
            )

    select = np.argmax(np.sum(masks, axis=(1,2)))
    mask = masks[select] * 255
    image = np.zeros(mask.shape+(3,), dtype=np.int32)
    image[:,:,0] = mask
    image[:,:,1] = mask
    image[:,:,2] = mask
    return image

