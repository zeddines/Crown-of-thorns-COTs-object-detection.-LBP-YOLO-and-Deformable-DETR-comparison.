#!/usr/bin/bash

git clone https://github.com/WongKinYiu/yolov7.git
#dos2unix patch0.patch
#unix2dos patch0.patch
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gIk9IG-UvlNt2h6ntmackshS1g03b2Zg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gIk9IG-UvlNt2h6ntmackshS1g03b2Zg" -O files/runs.zip && rm -rf /tmp/cookies.txt
unzip files/runs.zip -d runs
patch -p1 -b -r --binary --forward < patch0.patch -d yolov7/
python3 YOLO_preprocess.py --detr_p ../Deformable-DETR-Kaggle-Starfish-Detection/pre_proc --gbr_p ../Deformable-DETR-Kaggle-Starfish-Detection/tensorflow-great-barrier-reef
