1. Download tensorflow-great-barrier-reef to this directory
1. Follow the instructions in DETR and YOLO
2. Run scoring.py to get the best score from Deformable-DETR and YOLO

For Deformable-DETR
python3 scoring.py --detr_p predictions/detr_test_results.json --con_thres 0.2

For YOLO
python3 scoring.py --yolo_p predictions/yolo_test_results.json --con_thres 0.05

To generate images with bounding boxes, add --generate 
