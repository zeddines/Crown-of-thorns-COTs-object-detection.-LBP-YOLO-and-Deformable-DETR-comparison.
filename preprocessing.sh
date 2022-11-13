#!/bin/sh

mv tensorflow-great-barrier-reef Deformable-DETR-Kaggle-Starfish-Detection/files/
cd Deformable-DETR-Kaggle-Starfish-Detection
sh pre_proc/detr_preproc.sh
cd ../YOLO
if [ `uname` = "msys" ] ;
then
    unix2dos patch0.patch
elif [ `uname` = "cygwin" ];
then
    unix2dos patch0.patch
else
    dos2unix patch0.patch
fi
sh process.sh
cd ..