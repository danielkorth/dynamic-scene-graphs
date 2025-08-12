from dsg.utils.sam2_utils import (create_overlapping_subsets, detect_with_furthest, is_new_obj, mask_first_frame, 
                                save_sam_cv2, save_points_image_cv2_obj_id, make_video_from_visualizations, detect_with_cc, get_mask_from_points, save_obj_points, solve_overlap)
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from dsg.segment import SAM2Segmenter
import hydra
import os
from PIL import Image
import numpy as np
import torch

@hydra.main(config_path="../configs", config_name="sam2_reinit.yaml")
def main(cfg):
    # load images
    predictor = build_sam2_video_predictor(
        config_file = cfg.sam.model_cfg,
        ckpt_path = cfg.sam.sam2_checkpoint,
        device = "cuda"
    )

    auto_segmenter = hydra.utils.instantiate(cfg.sam)
    img_segmenter = SAM2ImagePredictor(auto_segmenter.sam2)

    # load first image
    first_image_path = "/local/home/dkorth/Projects/dynamic-scene-graphs/data/zed/occlusion_pen/images_undistorted_crop/left000000.png"
    img = Image.open(first_image_path)
    img_np = np.array(img)

    # auto_segmenter.segment(img_np)
    
    points = torch.tensor([[100, 100], [200, 200], [300, 300]], dtype=torch.float32, device=img_segmenter.device)
    img_segmenter.set_image(img_np)
    in_points = img_segmenter._transforms.transform_coords(points, normalize=True, orig_hw=(img_np.shape[0], img_np.shape[1]))
    in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
    in_points = in_points[:, None, :]
    in_labels = in_labels[:, None]
    masks, iou_predictions, low_res_masks = img_segmenter._predict(
        in_points,
        in_labels,
        multimask_output=True,
        return_logits=True,
    )

    # ONE STEP DEEPER
    sparse_embeddings, dense_embeddings = img_segmenter.model.sam_prompt_encoder(
        points=(in_points, in_labels),
        boxes=None,
        masks=None,
    )

    print(sparse_embeddings.shape, dense_embeddings.shape)

if __name__ == "__main__":
    main()