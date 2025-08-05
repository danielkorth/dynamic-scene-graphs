import os
import numpy as np
from dsg.utils.sam2_utils import mask_first_frame
from dsg.utils.tools import get_color_for_id
import cv2
import hydra
from ultralytics import YOLO

# Only sample x in [0, 0.5] and y in [0.5, 1.0] (bottom left quarter)
xs = np.linspace(0, 0.5, 10)
ys = np.linspace(0.5, 1.0, 10)
grid = np.array([[x, y] for y in ys for x in xs])  # shape (100, 2)
point_grids = [grid]

@hydra.main(config_path="../configs", config_name="sam2_reinit.yaml")
def main(cfg):
    # Hardcoded image paths
    image_paths = [
        f'{cfg.paths.data_dir}zed/kitchen1/images_undistorted_crop/left000000.png',
        f'{cfg.paths.data_dir}zed/kitchen2/images_undistorted_crop/left000000.png',
        f'{cfg.paths.data_dir}zed/office1/images_undistorted_crop/left000000.png',
        f'{cfg.paths.data_dir}zed/office2/images_undistorted_crop/left000000.png',
        f'{cfg.paths.data_dir}zed/office3/images_undistorted_crop/left000000.png',
        # Add more paths as needed
    ]

    # Load YOLOv8s model
    yolo_model = YOLO('yolov8s.pt')

    for i, img_path in enumerate(image_paths):
        print(f'Processing {img_path}')
        img = cv2.imread(img_path)
        if img is None:
            print(f'Could not load image: {img_path}')
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay_img = img_rgb.copy().astype(np.float32)
        segmenter = hydra.utils.instantiate(cfg.sam)

        masks = mask_first_frame(img_rgb, segmenter, viz=False)

        for j, mask in enumerate(masks):
            color = get_color_for_id(j)
            color_rgb = np.array(color) * 255
            mask_arr = mask['segmentation']
            if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
                mask_arr = mask_arr.squeeze(0)
            elif mask_arr.ndim == 3:
                mask_arr = np.any(mask_arr, axis=0)
            elif mask_arr.ndim != 2:
                print(f"Warning: Unexpected mask shape {mask_arr.shape} for object {j}, skipping overlay")
                continue
            if mask_arr.shape != overlay_img.shape[:2]:
                print(f"Warning: Mask shape {mask_arr.shape} doesn't match image shape {overlay_img.shape[:2]} for object {j}, skipping overlay")
                continue
            mask_bool = mask_arr.astype(bool)
            if np.any(mask_bool):
                overlay_img[mask_bool] = overlay_img[mask_bool] * 0.4 + color_rgb * 0.6

        result_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(cfg.output_folder, f"{os.path.basename(img_path).split('.')[0]}_{i}_overlay.png")
        cv2.imwrite(out_path, result_bgr)
        print(f'Saved overlay for {img_path} to {out_path}')

        # --- YOLO detection and overlay ---
        yolo_results = yolo_model(img_rgb)
        yolo_img = result_img.copy()
        for r in yolo_results:
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else []
            confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else []
            clss = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, 'cls') else []
            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                label = r.names[cls] if hasattr(r, 'names') else str(cls)
                cv2.rectangle(yolo_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(yolo_img, f'{label} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        yolo_out_path = os.path.join(cfg.output_folder, f"{os.path.basename(img_path).split('.')[0]}_{i}_overlay_yolo.png")
        yolo_img_bgr = cv2.cvtColor(yolo_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(yolo_out_path, yolo_img_bgr)
        print(f'Saved YOLO+mask overlay for {img_path} to {yolo_out_path}')

if __name__ == "__main__":
    main()