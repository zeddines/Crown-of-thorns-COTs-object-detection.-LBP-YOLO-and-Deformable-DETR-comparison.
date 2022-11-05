#!/usr/bin/bash

patch --forward -d Deformable-DETR -p1 < files/patch0.patch
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1J0Wa6uc-X7EShMH6qOsckRaB8KvXZbn-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1J0Wa6uc-X7EShMH6qOsckRaB8KvXZbn-" -O files/models.tar.xz && rm -rf /tmp/cookies.txt
#Use this instead if want full checkpoints
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bUa-xZhW6YfBnj2vA9ZWXdXSZEFgw4oR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bUa-xZhW6YfBnj2vA9ZWXdXSZEFgw4oR" -O files/models.tar.xz && rm -rf /tmp/cookies.txt
tar -xvf files/models.tar.xz -C Deformable-DETR
pip3 install --user -r pre_proc/requirements.txt
var1=$( pip3 show albumentations | grep 'Location:' | cut -d ':' -d ' ' -f 2)
var2="/albumentations/core"
var3=${var1}${var2}
cp files/bbox_utils.py $var3/bbox_utils.py
mv files/tensorflow-great-barrier-reef pre_proc
python3 pre_proc/detr_preproc.py
mv pre_proc/coco Deformable-DETR/data
