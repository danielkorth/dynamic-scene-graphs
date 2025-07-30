#! /bin/bash

python src/test_mask_first_frame.py -m \
    sam.points_per_batch=64 \
    sam.pred_iou_thresh=0.4,0.6,0.8 \
    sam.box_nms_thresh=0.5,0.7,0.9 \
    sam.points_per_side=32,40,48 \
    sam.use_m2m=False,True
