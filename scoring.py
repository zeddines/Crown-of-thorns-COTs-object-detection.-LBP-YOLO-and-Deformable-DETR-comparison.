from itertools import chain
import json
import os
import cv2 as cv
import pandas as pd
#from YOLO.YOLO_preprocess import split_annotate_with_without_class, convert_annotaions_to_dicts
import numpy as np
from copy import deepcopy
import argparse
import pathlib
import re
import shutil
#params
img_height = 720
img_width = 1280
s_iou_threshold = 0.3
f_iou_threshold = 0.8
iou_increment = 0.05
beta = 2

DETR_result_path = "detr_f2.txt"
YOLO_result_path = "yolo_f2.txt"

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


def judge_detections(gt, pred):
    F2 = TP = FP = FN = i = 0
    full_results = ""
    if (gt.keys() != pred.keys()):
        print("no of images in ground truth and predictions don't match")
        return
    for iou_threshold in np.linspace(s_iou_threshold, f_iou_threshold, int((f_iou_threshold-s_iou_threshold)/iou_increment) + 1):
        for img_id in gt.keys():
        #for img_id in ["0-35"]:
            bound_gt = deepcopy(gt[img_id])
            bound_pred = deepcopy(pred[img_id])
            #print(bound_gt)
            #print(bound_pred)
            tp, fp, fn = find_f2_single_image(bound_gt, bound_pred, iou_threshold)
            TP+=tp; FP += fp; FN+=fn
        print(f"TP = {TP}, iou threshold = {iou_threshold}")
        print(f"FP = {FP}, iou threshold = {iou_threshold}")
        print(f"FN = {FN}, iou threshold = {iou_threshold}")
        print(f"F2_iou = {((1+beta**2)*TP)/((1+beta**2)*TP+(beta**2)*FN+FP)}, iou threshold = {iou_threshold}")
        full_results+=f"TP = {TP}, iou threshold = {iou_threshold}\n"
        full_results+=f"FP = {FP}, iou threshold = {iou_threshold}\n"
        full_results+=f"FN = {FN}, iou threshold = {iou_threshold}\n"
        full_results+=f"F2_iou = {((1+beta**2)*TP)/((1+beta**2)*TP+(beta**2)*FN+FP)}, iou threshold = {iou_threshold}\n"
        F2 += ((1+beta**2)*TP)/((1+beta**2)*TP+(beta**2)*FN+FP)
        TP = FN = FP = 0
        i += 1
    return F2/i, full_results

#bounding box is a list of dict
#keys = x, y, width, height, score, note x, y width and height are absolute to image
def find_f2_single_image(bound_gt, bound_pred, threshold):   
    bound_pred.sort(key = lambda x: x["score"], reverse = True)
    TP = FP = FN = 0
    for box in bound_pred:
        if len(bound_gt) != 0:
            iou_list = list(map(lambda b: find_iou(box, b), bound_gt))
            
            max_iou = max(iou_list)
            i = iou_list.index(max_iou)
            if (max_iou > threshold):
                TP += 1
            else:
                FP += 1
            bound_gt.pop(i)
        ##assuming if there is no more gt boxes and still have pred boxes, iou = 0 and it counted as FP
        else:
            FP += 1
    FN += len(bound_gt)
    return TP, FP, FN

def find_iou(bb_pred, bb_gt):
    width_intersect = max(min(bb_pred["x"]+bb_pred["width"], bb_gt["x"]+bb_gt["width"]) - max(bb_pred["x"], bb_gt["x"]), 0)
    height_intersect = max(min(bb_pred["y"]+bb_pred["height"], bb_gt["y"]+bb_gt["height"]) - max(bb_pred["y"], bb_gt["y"]), 0)
    return width_intersect*height_intersect/(bb_pred["width"]*bb_pred["height"]+bb_gt["width"]*bb_gt["height"]-width_intersect*height_intersect)

def process_gt(img_id_list):
    df = pd.read_csv(os.path.join("tensorflow-great-barrier-reef", 'train.csv'), converters={'annotations': lambda x: convert_annotaions_to_dicts(x)})
    wo, w = split_annotate_with_without_class(df)
    gt = wo | w    
    
    gt_validation = {}
    for img_id, bb_list in gt.items():
        #filter out photos not in validatoin set
        if img_id in img_id_list:
            bb_new_list = [dict(x = d["x"], y = d["y"], width = d["width"], height = d["height"]) for d in bb_list]
            gt_validation[img_id] = bb_new_list
    return gt_validation  

def process_DETR(DETR_predictions_path):
    with open(DETR_predictions_path) as js_file:
        prediction = json.load(js_file)
    for k, v in prediction.items():
        bb_list = v["bboxes"]
        score_list = v["scores"]
        bb_new_list = [dict(x = i[0], y = i[1], width = i[2]-i[0], height = i[3]-i[1], score = score_list[c])  for c, i in enumerate(bb_list)]
        prediction[k] = bb_new_list

    return prediction

def process_YOLO(YOLO_predictions_path):
    with open(YOLO_predictions_path) as js_file:
        prediction = json.load(js_file)
    new_pred = {d["image_id"]:[] for d in prediction}
    for d in prediction:
        img_id = d["image_id"]
        bb = d["bbox"]
        bb = dict(x = d["bbox"][0], y = d["bbox"][1], width = d["bbox"][2], height = d["bbox"][3], score = d["score"])
        new_pred[d["image_id"]].append(bb)

    return new_pred

#red bounding box is ground truth, blue is prediction
def generate_bounding_box_images(results, gt, new_folder_name):
    if (os.path.isdir(new_folder_name)):
        shutil.rmtree(new_folder_name)
    os.mkdir(new_folder_name)
    for img_id, bb_list in results.items():
        img_path = os.path.join("tensorflow-great-barrier-reef","train_images", f"video_{img_id.split('-')[0]}", f"{img_id.split('-')[1]}.jpg")
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        for d in bb_list:
            #image, start_point, end_point, color, thickness
            img = cv.rectangle(img, (int(d["x"]), int(d["y"])), (int(d["x"]+d["width"]), int(d["y"]+d["height"])), (255, 0, 0), 2)
        for d in gt[img_id]:
            img = cv.rectangle(img, (int(d["x"]), int(d["y"])), (int(d["x"]+d["width"]), int(d["y"]+d["height"])), (0, 0, 255), 2)
        cv.imwrite(os.path.join(new_folder_name, f"{img_id}.jpg"), img)


def filter_bb_score(results, score):
    for img_id, bb_list in results.items():
        new_bb_list = [i for i in bb_list if i["score"] >= score]
        results[img_id] = new_bb_list
    return results

#extract results and ground truth and preprocessing format, both gt and pred are dictionary of key = img_id, value = list of bounding boxes 

parser = argparse.ArgumentParser(description = "generate result and f2 score")
parser.add_argument("--yolo_p", type = pathlib.Path, default = None)
parser.add_argument("--con_thres", type = float, default = 0.01)
parser.add_argument("--detr_p", type = pathlib.Path, default = None)
parser.add_argument("--generate", action = "store_true")

args = parser.parse_args()

if args.detr_p != None: 
    pred = process_DETR(args.detr_p)
    gt = process_gt(pred.keys())
    pred = filter_bb_score(pred, args.con_thres)
    if args.generate:
        generate_bounding_box_images(pred, gt, "DETR bounding results")
    s, full_results = judge_detections(gt, pred)
    print(f"f2-score = {s}")
    full_results+=f"f2-score = {s}"
    f = open(DETR_result_path, "w")
    f.write(full_results)
    f.close()


if args.yolo_p != None:
    pred = process_YOLO(args.yolo_p)
    gt = process_gt(pred.keys())
    pred = filter_bb_score(pred, args.con_thres)
    ###testimg
    #print(pred["2-5735"])
    #pred = {"2-5735":pred["2-5735"]}
    #gt = {"2-5735":gt["2-5735"]}
    ###
    if args.generate:
        generate_bounding_box_images(pred, gt, "YOLO bounding results")
    s, full_results = judge_detections(gt, pred)
    full_results+=f"f2-score = {s}"
    print(f"f2-score = {s}")
    f = open(YOLO_result_path, "w")
    f.write(full_results)
    f.close()