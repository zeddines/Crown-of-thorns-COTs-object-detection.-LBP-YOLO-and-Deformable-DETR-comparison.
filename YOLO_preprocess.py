import pandas as pd
import re
import json
import os
#params
data_path = 'tensorflow-great-barrier-reef'
annotation_file_path = "training-annotation"
#grid_size = 80
#num_bounding_box = 2
img_height = 720
img_width = 1280
label_no = 0

def convert_annotaions_to_dicts(annotaion):
    return [json.loads(item) for item in re.findall(r'{.*?}', annotaion.replace("'", '"'))]

def remove_img_without_classes(df):
    out_dict = {}
    for i in range(len(df.index)):
        if df['annotations'][i]:
            out_dict[df['image_id'][i]] = df['annotations'][i]
    return out_dict


def create_training_annotation(data):
    if not os.path.isdir(annotation_file_path):
        os.makedirs(annotation_file_path)

    for image_id, annotations in data.items():
        f = open(f"{os.path.join(annotation_file_path,image_id)}.txt", "w+")
        #[y][x], y is grid rows, x is grid columns
        #output_dict[image_id] = [[[] for x in range(0, math.floor(img_width/grid_size))] for y in range(0, math.floor(img_height/grid_size))]
        for object in annotations:
            x, y, width, height = object.values()
            xc, yc, rw, rh= find_object_params(x, y, width, height)
            f.write(f"{label_no} {xc} {yc} {rw} {rh}\n")

def find_object_params(x, y, width, height):
    return  (x+width/2)/img_width, (y+height/2)/img_height,width/img_width, height/img_height


df = pd.read_csv(os.path.join(data_path, 'train.csv'), converters={'annotations': lambda x: convert_annotaions_to_dicts(x)})

training_data = remove_img_without_classes(df)
print(training_data)
create_training_annotation(training_data)