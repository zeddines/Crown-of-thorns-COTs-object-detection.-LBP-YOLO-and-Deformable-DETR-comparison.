import json
import random
import re
import pandas as pd
import os
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
from PIL import Image
import albumentations as A

#Directory Params
parent_dir = os.path.join(os.path.abspath(os.getcwd()), 'pre_proc')
data_path = os.path.join(parent_dir, 'tensorflow-great-barrier-reef')
root_path = os.path.join(parent_dir, 'coco')
train_img_path = os.path.join(root_path, 'train2017')
val_img_path = os.path.join(root_path, 'val2017')
annotations_path = os.path.join(root_path, 'annotations')
log_path = os.path.join(parent_dir, 'log')
augmented_path = os.path.join(parent_dir, 'augmented')

def convert_annotaions_to_dicts(annotaion):
    return [json.loads(item) for item in re.findall(r'{.*?}', annotaion.replace("'", '"'))]

def split_annotations_class(df):
    with_class_annotations = {}
    without_class_annotations = {}
    for i in range(len(df['annotations'])):
        if df['annotations'][i]:
            with_class_annotations[df['image_id'][i]] = df['annotations'][i]
        else:
            without_class_annotations[df['image_id'][i]] = df['annotations'][i]
    return with_class_annotations, without_class_annotations

def augment_imgs_with_class(img_id_list, annotations_dict):
    id_count = 0
    augmented_dict = {}
    transforms = [HorizontalFlip(), VerticalFlip(), DiagonalFlip()]
    for transform in transforms:
        for id in img_id_list:
            vid_id, img_id = id.split('-')[0], id.split('-')[1]
            img = Image.open(os.path.join(data_path, 'train_images', 'video_' + vid_id, img_id + '.jpg'))
            bboxes = []
            for annotation in annotations_dict[id]:
                bboxes.append([annotation['x'], annotation['y'], annotation['width'], annotation['height'], 'cot-starfish'])
            aug_id = 'aug-' + str(id_count)
            augmented = transform(image=np.array(img), bboxes=bboxes)
            augmented_dict[aug_id] = []
            for bbox in augmented['bboxes']:
                augmented_dict[aug_id].append({'x': bbox[0], 'y': bbox[1], 'width': bbox[2], 'height': bbox[3]})
            (Image.fromarray(augmented['image'])).save(os.path.join(augmented_path, aug_id + '.jpg'))
            id_count += 1
    return augmented_dict

def remap_img_id(img_id_list):
    new_img_id_map = {}
    count = 0
    for id in img_id_list:
        new_img_id_map[id] = count
        count += 1
    return new_img_id_map

def shuffle_img_id_dict(img_id_dict):
    keys = list(img_id_dict.keys())
    random.shuffle(keys)
    new_img_id_dict = dict()
    for key in keys:
        new_img_id_dict.update({key: img_id_dict[key]})
    return new_img_id_dict

def create_augment_train_val_images(img_id_list_with_class, img_id_list_without_class, with_class_annotations):

    train_img_id_with_class, val_img_id_with_class = train_test_split(img_id_list_with_class, train_size=0.8, random_state=42)
    train_img_id_without_class, val_img_id_without_class = train_test_split(img_id_list_without_class, train_size=0.8, random_state=42)

    augmented_annotations_dict = augment_imgs_with_class(train_img_id_with_class, with_class_annotations)
    train_img_id_map = remap_img_id(train_img_id_with_class + train_img_id_without_class + list(augmented_annotations_dict.keys()))
    val_img_id_map = remap_img_id(val_img_id_with_class + val_img_id_without_class)

    for id in train_img_id_map:
        img_name = None
        vid_id, img_id = id.split("-")[0], id.split("-")[1]
        if vid_id == 'aug':
            img_name = os.path.join(augmented_path, str(id) + '.jpg')
        else:
            img_name = os.path.join(data_path, 'train_images', 'video_' + str(vid_id), str(img_id) + '.jpg')
        new_img_name = os.path.join(train_img_path, str(train_img_id_map[id]) + '.jpg')
        shutil.copy(img_name, new_img_name)

    for id in val_img_id_map:
        vid_id, img_id = id.split("-")[0], id.split("-")[1]
        img_name = os.path.join(data_path, 'train_images', 'video_' + str(vid_id), str(img_id) + '.jpg')
        new_img_name = os.path.join(val_img_path, str(val_img_id_map[id]) + '.jpg')
        shutil.copy(img_name, new_img_name)

    return {y: x for x, y in train_img_id_map.items()}, {y: x for x, y in val_img_id_map.items()}, augmented_annotations_dict
        
def create_coco_format(annotations_dict, img_id_map, mode):

    img_path = None
    save_path = None
    if mode == 'train':
        img_path = train_img_path
        save_path = os.path.join(annotations_path, 'instances_train2017.json')
    elif mode == 'val':
        img_path = val_img_path
        save_path = os.path.join(annotations_path, 'instances_val2017.json')
    else:
        return

    coco = Coco()
    coco.add_category(CocoCategory(id=0, name='crown-of-thorns starfish'))

    for img_id in img_id_map:
        path_to_img = os.path.join(img_path, str(img_id) + '.jpg')
        width, height = Image.open(path_to_img).size
        coco_image = CocoImage(file_name=str(img_id) + '.jpg', height=height, width=width)
        for annotation in annotations_dict[img_id_map[img_id]]:
            coco_image.add_annotation(
                CocoAnnotation(
                bbox=[annotation['x'], annotation['y'], annotation['width'], annotation['height']],
                category_id=0,
                category_name='crown-of-thorns starfish',
                image_id=img_id
                )
            )
        coco.add_image(coco_image)
    
    save_json(data=coco.json, save_path=save_path)

def create_folders():
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(train_img_path):
        os.makedirs(train_img_path)
    if not os.path.exists(val_img_path):
        os.makedirs(val_img_path)
    if not os.path.exists(annotations_path):
        os.makedirs(annotations_path)
    if not os.path.exists(augmented_path):
        os.makedirs(augmented_path)


def HorizontalFlip():
    return A.Compose(
        [A.HorizontalFlip(always_apply=True)],
        bbox_params=A.BboxParams(format='coco')
    )

def VerticalFlip():
    return A.Compose(
        [A.VerticalFlip(always_apply=True)],
        bbox_params=A.BboxParams(format='coco')
    )

def DiagonalFlip():
    return A.Compose([
        A.HorizontalFlip(always_apply=True),
        A.VerticalFlip(always_apply=True)
    ], bbox_params=A.BboxParams(format='coco'))

def logs(train_img_id_map, val_img_id_map, augmented_annotations, annotations_dict):
    with open(os.path.join(log_path, 'train_img_id_map.json'), 'w') as f1:
        json.dump(train_img_id_map, f1)
    with open(os.path.join(log_path, 'val_img_id_map.json'), 'w') as f2:
        json.dump(val_img_id_map, f2)
    with open(os.path.join(log_path, 'augmented_annotations.json'), 'w') as f3:
        json.dump(augmented_annotations, f3)
    print("Total Images = " + str(len(annotations_dict)))
    print("Total Augmented Images = " + str(len(augmented_annotations)))
    print("Total Train Images = " + str(len(train_img_id_map)))
    print("Total Validation Images = " + str(len(val_img_id_map)))

def preprocess(df):
    create_folders()
    with_class_annotations, without_class_annotations = split_annotations_class(df)
    train_img_id_map, val_img_id_map, augmented_annotations = create_augment_train_val_images(list(with_class_annotations.keys()), list(without_class_annotations.keys()), with_class_annotations)
    annotations_dict = with_class_annotations | without_class_annotations | augmented_annotations
    create_coco_format(annotations_dict, shuffle_img_id_dict(train_img_id_map), 'train')
    create_coco_format(annotations_dict, val_img_id_map, 'val')
    logs(train_img_id_map, val_img_id_map, augmented_annotations, annotations_dict)

df = pd.read_csv(os.path.join(data_path, 'train.csv'), converters={'annotations': lambda x: convert_annotaions_to_dicts(x)})
preprocess(df)