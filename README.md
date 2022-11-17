1. Download tensorflow-great-barrier-reef to this directory
2. Run preprocess.sh (if you want to test YOLO and Deformable DETR again, the script will clone the repo and download the requirements needed. After that simply follow the official yolov7 github page and Deformable DETR page for testing) --optional

3. Run scoring.py to get the best score from Deformable-DETR and YOLO. 

For Deformable-DETR
python3 scoring.py --detr_p predictions/detr_test_results.json --con_thres 0.2

For YOLO
python3 scoring.py --yolo_p predictions/yolo_test_results.json --con_thres 0.05

To generate images with bounding boxes, add --generate 


4. To get image enhancement images, place image_enhance.py in tensorflow-great-barrier-reef and run it.

