import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def show_anns(mask_color_pairs, ax=None, borders=True):
    if len(mask_color_pairs) == 0:
        return
    if ax is None:
        ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((mask_color_pairs[0][0]['segmentation'].shape[0], mask_color_pairs[0][0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    # Sort by area, but keep color pairing
    sorted_pairs = sorted(mask_color_pairs, key=lambda pair: float(pair[0]['area']), reverse=True)
    for ann, color in sorted_pairs:
        m = ann['segmentation']
        color_mask = np.concatenate([color, [0.5]])
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

    def visualize_masks(self, image: np.ndarray, masks: List[dict], only_masks: bool = False, show_bbox: bool = False, show_points: bool = False, show_stability: bool = False) -> None:
        """
        Visualize masks on the image with optional overlays.
        Args:
            image: The input image (HWC, np.ndarray)
            masks: List of mask dicts (as output by segment())
            only_masks: If True, show only the masks, not the image
            show_bbox: If True, draw bounding boxes for each mask
            show_points: If True, draw point coordinates for each mask
            show_stability: If True, write the stability score for each mask
        """
        import matplotlib.patches as mpatches
        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        if not only_masks:
            ax.imshow(image)
        else:
            ax.imshow(np.zeros_like(image))
        n = len(masks)
        rng = np.random.default_rng(42)  # fixed seed for reproducibility
        colors = rng.random((n, 3))
        mask_color_pairs = [(masks[i], colors[i]) for i in range(n)]
        show_anns(mask_color_pairs, ax=ax)
        # For overlays, sort the pairs by area (same as show_anns)
        sorted_pairs = sorted(mask_color_pairs, key=lambda pair: float(pair[0]['area']), reverse=True)
        for ann, color in sorted_pairs:
            # Draw bbox
            if show_bbox and 'bbox' in ann:
                x, y, w, h = ann['bbox']
                rect = mpatches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', alpha=0.9)
                ax.add_patch(rect)
            # Draw points
            if show_points and 'point_coords' in ann:
                pts = np.array(ann['point_coords'])
                ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=80, marker='o', edgecolors='white', linewidths=2, zorder=10)
            # Write stability score
            if show_stability and 'stability_score' in ann:
                # Place text near top-left of bbox if available, else near first point
                if 'bbox' in ann:
                    x, y, _, _ = ann['bbox']
                    tx, ty = x, y - 5
                elif 'point_coords' in ann:
                    tx, ty = ann['point_coords'][0]
                else:
                    tx, ty = 0, 0
                ax.text(tx, ty, f"Stab: {ann['stability_score']:.3f}", color='yellow', fontsize=12, weight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'), zorder=20)
        ax.axis('off')
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
    