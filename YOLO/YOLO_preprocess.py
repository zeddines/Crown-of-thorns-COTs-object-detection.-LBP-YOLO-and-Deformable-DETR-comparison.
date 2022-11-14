import pandas as pd
import re
import json
import os
from sklearn.model_selection import train_test_split
import shutil
import argparse
import pathlib

#params
train_folder = os.path.join("yolov7", "data", "train")
val_folder = os.path.join("yolov7", "data", "val")
train_split = 0.8


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

#train val is dict of train_list and val_list
def split_annotate_test_train(annotate, train_val = None):
    if (train_val == None):
        s = pd.Series(annotate)
        training_annotate, test_annotate = [i.to_dict() for i in train_test_split(s, train_size = train_split)]
    else:
        training_annotate = {img_id: ann for img_id, ann in annotate.items() if img_id in train_val["train"]}
        test_annotate = {img_id: ann for img_id, ann in annotate.items() if img_id in train_val["val"]}
    return  training_annotate, test_annotate
        

def create_annotation_folder(ann_data, path):
    os.makedirs(os.path.join(path, "labels"))
    for image_id, annotations in ann_data.items():
        ann_path = os.path.join(path, "labels", image_id)
        f = open(f"{ann_path}.txt", "w+")
        for object in annotations:
            x, y, width, height = object.values()
            xc, yc, rw, rh= find_object_params(x, y, width, height)
            f.write(f"{label_no} {xc} {yc} {rw} {rh}\n")

def create_img_folder(ann_data, path):
    os.makedirs(os.path.join(path, "images"))
    for img_id in ann_data.keys():
        x1, x2 = img_id.split("-")
        if (x1 == "aug"):
            img_file = os.path.join(detr_data_path, "augmented", f"{img_id}.jpg")
        else:
            vid_no, img_no = img_id.split("-")
            img_file = os.path.join(gbr_path, "train_images", f"video_{vid_no}", f"{img_no}.jpg")
        shutil.copy(img_file, os.path.join(path, "images", f"{img_id}.jpg"))

def find_object_params(x, y, width, height):
    return  (x+width/2)/img_width, (y+height/2)/img_height,width/img_width, height/img_height

def create_data_folders(train_ann, test_ann, train_folder, test_folder):
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    create_annotation_folder(train_ann, train_folder)
    create_annotation_folder(test_ann, test_folder)
    create_img_folder(train_ann, train_folder)
    create_img_folder(test_ann, test_folder)

parser = argparse.ArgumentParser()
parser.add_argument("--gbr_p", type = pathlib.Path)
parser.add_argument("--detr_p", type = pathlib.Path)
args = parser.parse_args()

#gbr_path = os.path.join("files", "tensorflow-great-barrier-reef")
#detr_data_path = os.path.join("files", "data_new")
gbr_path = args.gbr_p
detr_data_path = args.detr_p


df = pd.read_csv(os.path.join(gbr_path, 'train.csv'), converters={'annotations': lambda x: convert_annotaions_to_dicts(x)})

annotate_wo_class, annotate_w_class = split_annotate_with_without_class(df)
"""
train_w_class, test_w_class = split_annotate_test_train(annotate = annotate_w_class)
train_wo_class, test_wo_class = split_annotate_test_train(annotate = annotate_wo_class)

#change ratio of positive and negative images for training if needed
train_ann = train_w_class | train_wo_class
test_ann = test_w_class | test_wo_class

create_data_folders(train_ann, test_ann, "train", "val")"""
annotations_original = annotate_w_class | annotate_wo_class 
f = open(os.path.join(detr_data_path, "log", "augmented_annotations.json"))
annotations_augmented = json.load(f)
annotations = annotations_original | annotations_augmented
f = open(os.path.join(detr_data_path, "log", "train_img_id_map.json"))
train_img_id = json.load(f).values()
f = open(os.path.join(detr_data_path, "log", "val_img_id_map.json"))
val_img_id = json.load(f).values()

train_ann, test_ann = split_annotate_test_train(annotations, {"train":train_img_id, "val":val_img_id})
create_data_folders(train_ann, test_ann, train_folder, val_folder)

