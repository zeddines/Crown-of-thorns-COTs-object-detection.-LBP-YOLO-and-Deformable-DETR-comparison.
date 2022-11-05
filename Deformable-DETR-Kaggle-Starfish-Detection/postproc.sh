#!/usr/bin/bash

cp -r Deformable-DETR/val_results post_proc
cp pre_proc/log/val_img_id_map.json post_proc
python3 post_proc/detr_postproc.py