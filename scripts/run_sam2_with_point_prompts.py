"""This module implements SAM2 with point prompts for single images.
"""
import argparse
import cv2
import json
import numpy as np
import os
import torch
from dotenv import load_dotenv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def args_parse():
    """Parses command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, help='Path to the image which is segmented')
    parser.add_argument('--point-prompt', type=str, help='Point prompts as string of lists')
    parser.add_argument('--output-dir', type=str, help='Path where the output masks will be saved to')
    parser.add_argument('--alpha', type=float, help='Alpha value for overlay mask', default=0.2)
    return parser.parse_args()

def main(img_path: str, point_prompt: str, output_dir: str=None, alpha: float=0.2):
    predictor = SAM2ImagePredictor(build_sam2(os.getenv('CFG'), os.getenv('CKPT')))

    img = np.array(Image.open(img_path).convert('RGB'))
    predictor.set_image(img)

    point_coords = np.array(json.loads(point_prompt))
    point_labels = np.ones(point_coords.shape[0])  # 0 -> substract, 1 -> add
    for i, p in enumerate(point_coords):
        masks, ious, lowres_logits = predictor.predict(
            point_coords=p[None, :],
            point_labels=np.array([point_labels[i]]),
            box=None,
            mask_input=None,
            multimask_output=False,
        )

        final_mask = masks[0] * 255.0
        overlay = img.copy()
        overlay[final_mask == 255] = (
            np.array((0, 255, 0), dtype=np.uint8) * alpha
            + overlay[final_mask == 255].astype(np.uint8) * (1 - alpha)
        ).astype(np.uint8)
        if output_dir:
            cv2.imwrite(f'{output_dir}/mask_{str(i).zfill(6)}.png', final_mask.astype(np.uint8))
            cv2.imwrite(f'{output_dir}/mask_{str(i).zfill(6)}.png', overlay)

if __name__ == '__main__':
    load_dotenv()
    args = args_parse()
    cfg = vars(args)
    main(**cfg)
