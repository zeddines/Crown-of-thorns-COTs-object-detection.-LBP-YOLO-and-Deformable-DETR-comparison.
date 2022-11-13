from YOLO_preprocess import data_path, convert_annotaions_to_dicts, split_annotate_with_without_class, convert_annotaions_to_dicts
import pandas as pd
import re
import json
import os
from sklearn.model_selection import train_test_split
import shutil
import cv2 as cv

df = pd.read_csv(os.path.join(data_path, 'train.csv'), converters={'annotations': lambda x: convert_annotaions_to_dicts(x)})
_, ann_w_class = split_annotate_with_without_class(df)

os.makedirs("cropped-cots")

for img_id in ann_w_class.keys():
        vid_no, img_no = img_id.split("-")
        img_file = os.path.join(data_path, "train_images", f"video_{vid_no}", f"{img_no}.jpg")
        img = cv.imread(img_file)
        for c, i in enumerate(ann_w_class[img_id]):
            y_s = int(i["y"])
            y_e = y_s + int(i["height"])
            x_s = int(i["x"])
            x_e = x_s + int(i["width"])
            img_cropped = img[y_s:y_e, x_s:x_e]
            cv.imwrite(os.path.join("cropped-cots", f"{img_id}-{str(c)}.jpg"), img_cropped)



        


