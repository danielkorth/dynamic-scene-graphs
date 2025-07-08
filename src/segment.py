import numpy as np
from typing import List
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def show_anns(anns, ax=None, borders=True):
    if len(anns) == 0:
        return
    if ax is None:
        ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


# mask_generator_2 = SAM2AutomaticMaskGenerator(
#     model=sam2,
#     points_per_side=64,
#     points_per_batch=128,
#     pred_iou_thresh=0.7,
#     stability_score_thresh=0.92,
#     stability_score_offset=0.7,
#     crop_n_layers=1,
#     box_nms_thresh=0.7,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=25.0,
#     use_m2m=True,
# )

class SAM2Segmenter:
    # segmenter = SAM2Segmenter(
    #     sam2_checkpoint="checkpoints/sam2.1/sam2.1_hiera_tiny.pt",
    #     model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml"
    # )
    # masks = segmenter.segment(image)
    def __init__(self, sam2_checkpoint: str, model_cfg: str, device: str = "cuda", apply_postprocessing: bool = False):
        self.sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=apply_postprocessing)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2)
        # self.mask_generator = SAM2AutomaticMaskGenerator(
        #     self.sam2, 
        #     points_per_side=64, 
        #     points_per_batch=128, 
        #     pred_iou_thresh=0.7, 
        #     stability_score_thresh=0.92, 
        #     stability_score_offset=0.7, 
        #     crop_n_layers=1, 
        #     box_nms_thresh=0.7, 
        #     crop_n_points_downscale_factor=2, 
        #     min_mask_region_area=25.0, 
        #     use_m2m=True
        # )

    def segment(self, image: np.ndarray) -> List[np.ndarray]:
        masks = self.mask_generator.generate(image)
        return masks

    def visualize_masks(self, image: np.ndarray, masks: List[np.ndarray], only_masks: bool = False) -> None:
        plt.figure(figsize=(20, 20))
        if not only_masks:
            plt.imshow(image)
        else:
            plt.imshow(np.zeros_like(image))
        show_anns(masks)
        plt.axis('off')
        plt.show()
    
    def get_masks_figure(self, image: np.ndarray, masks: List[np.ndarray], only_masks: bool = False):
        fig, ax = plt.subplots(figsize=(20, 20))
        if not only_masks:
            ax.imshow(image)
        else:
            ax.imshow(np.zeros_like(image))
        show_anns(masks, ax=ax)
        ax.axis('off')
        return fig
    