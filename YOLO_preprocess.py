import pandas as pd
import re
import json

def convert_annotaions_to_dicts(annotaion):
    return [json.loads(item) for item in re.findall(r'{.*?}', annotaion.replace("'", '"'))]

def remove_img_without_classes(df):
    out_dict = {}
    for i in range(len(df.index)):
        if df['annotations'][i]:
            out_dict[df['image_id'][i]] = df['annotations'][i]
    return out_dict
    
data_path = 'tensorflow-great-barrier-reef/'
df = pd.read_csv(data_path + 'train.csv', converters={'annotations': lambda x: convert_annotaions_to_dicts(x)})
print(remove_img_without_classes(df))