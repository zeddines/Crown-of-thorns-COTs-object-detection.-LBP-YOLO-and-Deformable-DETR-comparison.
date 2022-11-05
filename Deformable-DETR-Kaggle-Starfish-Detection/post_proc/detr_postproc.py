import json
import os

parent_dir = os.path.join(os.path.abspath(os.getcwd()), 'post_proc')
input_path = os.path.join(parent_dir, 'val_results')

def read_val_results():
    val_keys = None
    with open(os.path.join(parent_dir, 'val_img_id_map.json')) as f:
        val_keys = json.load(f)
    out_dict = {}
    for file in os.listdir(input_path):
        filename = os.fsdecode(file)
        with open(os.path.join(input_path, filename)) as f:
            out_dict[val_keys[filename.split(".")[0]]] = json.load(f)
    return out_dict

def postprocessing():
    results = read_val_results()
    with open(os.path.join(parent_dir, 'val_results.json'), 'w') as f:
        json.dump(results, f)
    
postprocessing()