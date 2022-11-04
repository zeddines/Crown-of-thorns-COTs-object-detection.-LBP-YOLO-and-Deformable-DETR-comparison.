from itertools import chain
import json
import os
import cv2 as cv
import pandas as pd
from YOLO_preprocess import split_annotate_with_without_class, convert_annotaions_to_dicts
import numpy as np
from copy import deepcopy
#params
img_height = 720
img_width = 1280
s_iou_threshold = 0.3
f_iou_threshold = 0.8
iou_increment = 0.05
beta = 2

DETR_predictions_path = "val_results.json"

def judge_detections(gt, pred):
    F2 = TP = FP = FN = i = 0
    if (gt.keys() != pred.keys()):
        print("no of images in ground truth and predictions don't match")
        return
    for iou_threshold in np.linspace(s_iou_threshold, f_iou_threshold, int((f_iou_threshold-s_iou_threshold)/iou_increment) + 1):
        for img_id in gt.keys():
        #for img_id in ["2-5735"]:
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
        F2 += ((1+beta**2)*TP)/((1+beta**2)*TP+(beta**2)*FN+FP)
        TP = FN = FP = 0
        i += 1
    return F2/i

#bounding box is a list of dict
#keys = cx, cy, width, height, score, note cx, cy width and height are absolute to image
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
    width_intersect = max(min(bb_pred["cx"]+bb_pred["width"]/2, bb_gt["cx"]+bb_gt["width"]/2) - max(bb_pred["cx"]-bb_pred["width"]/2, bb_gt["cx"]-bb_gt["width"]/2), 0)
    #print(width_intersect)
    height_intersect = max(min(bb_pred["cy"]+bb_pred["height"]/2, bb_gt["cy"]+bb_gt["height"]/2) - max(bb_pred["cy"]-bb_pred["height"]/2, bb_gt["cy"]-bb_gt["height"]/2), 0)
    #print(height_intersect)
    #print(width_intersect*height_intersect/(bb_pred["width"]*bb_pred["height"]+bb_gt["width"]*bb_gt["height"]))
    return width_intersect*height_intersect/(bb_pred["width"]*bb_pred["height"]+bb_gt["width"]*bb_gt["height"]-width_intersect*height_intersect)

def process_gt(img_id_list):
    df = pd.read_csv(os.path.join("tensorflow-great-barrier-reef", 'train.csv'), converters={'annotations': lambda x: convert_annotaions_to_dicts(x)})
    wo, w = split_annotate_with_without_class(df)
    gt = wo | w    
    
    gt_validation = {}
    for img_id, bb_list in gt.items():
        #filter out photos not in validatoin set
        if img_id in img_id_list:
            bb_new_list = [dict(cx = d["x"]+int(d["width"]/2), cy = d["y"]+int(d["height"]/2), width = d["width"], height = d["height"]) for d in bb_list]
            gt_validation[img_id] = bb_new_list
    return gt_validation  

def process_DETR():
    with open(DETR_predictions_path) as js_file:
        prediction = json.load(js_file)
    for k, v in prediction.items():
        bb_list = v["bboxes"]
        score_list = v["scores"]
        bb_new_list = [dict(cx = (i[0]+i[2])/2, cy = (i[1]+i[3])/2, width = i[2]-i[0], height = i[3]-i[1], score = score_list[c])  for c, i in enumerate(bb_list)]
        prediction[k] = bb_new_list
    
    return prediction

#red bounding box is ground truth, blue is prediction
def generate_bounding_box_images(results, gt, new_folder_name):
    if (os.path.isdir(new_folder_name)):
        return 
    os.mkdir(new_folder_name)
    for img_id, bb_list in results.items():
        img_path = os.path.join("tensorflow-great-barrier-reef","train_images", f"video_{img_id.split('-')[0]}", f"{img_id.split('-')[1]}.jpg")
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        for d in bb_list:
            #image, start_point, end_point, color, thickness
            img = cv.rectangle(img, (int(d["cx"]-d["width"]/2), int(d["cy"]-d["height"]/2)), (int(d["cx"]+d["width"]/2), int(d["cy"]+d["height"]/2)), (255, 0, 0), 2)
        for d in gt[img_id]:
            img = cv.rectangle(img, (int(d["cx"]-d["width"]/2), int(d["cy"]-d["height"]/2)), (int(d["cx"]+d["width"]/2), int(d["cy"]+d["height"]/2)), (0, 0, 255), 2)
        cv.imwrite(os.path.join("DETR bounding results",f"{img_id}.jpg"), img)

#extract results and ground truth and preprocessing format, both gt and pred are dictionary of key = img_id, value = list of bounding boxes 

pred = process_DETR()
print(np.linspace(s_iou_threshold, f_iou_threshold, int((f_iou_threshold-s_iou_threshold)/iou_increment) + 1))
gt = process_gt(pred.keys())
generate_bounding_box_images(pred, gt,"DETR bounding results")
s = judge_detections(gt, pred)
print(f"f2-score = {s}")