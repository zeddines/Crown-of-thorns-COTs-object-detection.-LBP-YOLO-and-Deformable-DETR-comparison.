import pandas as pd
import re
import json
import os
from sklearn.model_selection import train_test_split
import shutil

#params
data_path = 'tensorflow-great-barrier-reef'
training_ann_file_path = "training-annotation-YOLO"
test_ann_fille_path = "test-annotation-YOLO"
training_img_file_path = "training-images-YOLO"
test_img_file_path = "test-images-YOLO"
train_split = 0.8
#train_w_wo_class_split = 0.5

img_height = 720
img_width = 1280
label_no = 0

def convert_annotaions_to_dicts(annotaion):
    return [json.loads(item) for item in re.findall(r'{.*?}', annotaion.replace("'", '"'))]

def split_annotate_with_without_class(df):
    out_dict1 = {}
    out_dict2 = {}
    for i in range(len(df.index)):
        if df['annotations'][i]:
            out_dict2[df['image_id'][i]] = df['annotations'][i]
        else:
            out_dict1[df['image_id'][i]] = []
    return out_dict1, out_dict2

def split_annotate_test_train(annotate):
    s = pd.Series(annotate)
    training_annotate, test_annotate = [i.to_dict() for i in train_test_split(s, train_size = train_split)]
    return training_annotate, test_annotate

def create_annotation_folder(ann_data, foldername):
    if os.path.isdir(foldername):
        return
    os.makedirs(foldername)
    for image_id, annotations in ann_data.items():
        f = open(f"{os.path.join(foldername,image_id)}.txt", "w+")
        for object in annotations:
            x, y, width, height = object.values()
            xc, yc, rw, rh= find_object_params(x, y, width, height)
            f.write(f"{label_no} {xc} {yc} {rw} {rh}\n")

def create_img_folder(ann_data, foldername):
    if os.path.isdir(foldername):
        return
    os.makedirs(foldername)
    for img_id in ann_data.keys():
        vid_no, img_no = img_id.split("-")
        img_file = os.path.join(data_path, "train_images", f"video_{vid_no}", f"{img_no}.jpg")
        shutil.copy(img_file, os.path.join(foldername, f"{img_id}.jpg"))

def find_object_params(x, y, width, height):
    return  (x+width/2)/img_width, (y+height/2)/img_height,width/img_width, height/img_height

def create_data_folders(train_ann, test_ann):
    create_annotation_folder(train_ann, training_ann_file_path)
    create_annotation_folder(test_ann, test_ann_fille_path)
    create_img_folder(train_ann, training_img_file_path)
    create_img_folder(test_ann, test_img_file_path)

df = pd.read_csv(os.path.join(data_path, 'train.csv'), converters={'annotations': lambda x: convert_annotaions_to_dicts(x)})

annotate_wo_class, annotate_w_class = split_annotate_with_without_class(df)
train_w_class, test_w_class = split_annotate_test_train(annotate_w_class)
train_wo_class, test_wo_class = split_annotate_test_train(annotate_wo_class)

#change ratio of positive and negative images for training if needed
train_ann = train_w_class | train_wo_class
test_ann = test_w_class | test_wo_class

create_data_folders(train_ann, test_ann)