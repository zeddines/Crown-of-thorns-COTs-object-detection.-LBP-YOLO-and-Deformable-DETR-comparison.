from itertools import chain
#params
img_height = 720
img_width = 1280
s_iou_threshold = 0.3
f_iou_threshold = 0.8
iou_increment = 0.05
beta = 2

def judge_detections(gt, pred):
    F2, TP, FP, FN, i = 0
    if (gt.keys() != pred.keys()):
        print("no of images in ground truth and predictions don't match")
        return
    for iou_threshold in chain(range(s_iou_threshold, f_iou_threshold, iou_increment), f_iou_threshold):
        for img_id in gt.keys():
            bound_gt = gt[img_id].copy()
            bound_pred = pred[img_id].copy()
            tp, fp, fn = find_f2_single_image(bound_gt, bound_pred, iou_threshold)
            TP+=tp; FP += fp; FN+=fn
        F2 += ((1+beta^2)*TP)/((1+beta^2)*TP+(beta^2)*FN*FP)
        TP, FN, FP = 0
        i += 1
    return F2/i

#bounding box is a list of dict
#keys = cx, cy, width, height, confidence score, note cx, cy width and height are absolute to image
def find_f2_single_image(bound_gt, bound_pred, threshold):   
    bound_pred.sort(key = lambda x: x["confidence score"], reverse = True)
    TP, FP, FN = 0
    for box in bound_pred:
        iou_list = map(lambda b: find_iou(box, b), bound_gt)
        max_iou = max(iou_list)
        i = iou_list.index(max_iou)
        if (max_iou > threshold):
            TP += 1
        else:
            FP += 1
        bound_gt.pop(i)
    FN += len(bound_gt)
    return TP, FP, FN

        

def find_iou(bb_pred, bb_gt):
    width_intersect = max(min(bb_pred["cx"]+bb_pred["width"]/2, bb_gt["cx"]+bb_gt["width"]/2) - max(bb_pred["cx"]-bb_pred["width"]/2, bb_gt["cx"]-bb_gt["width"]/2), 0)
    height_intersect = max(min(bb_pred["cy"]+bb_pred["height"]/2, bb_gt["cy"]+bb_gt["height"]/2) - max(bb_pred["cy"]-bb_pred["height"]/2, bb_gt["cy"]-bb_gt["height"]/2), 0)
    return width_intersect*height_intersect/(bb_pred["width"]*bb_pred["height"]+bb_gt["width"]*bb_gt["height"])

#extract results and ground truth and preprocessing format, both gt and pred are dictionary of key = img_id, value = list of bounding boxes 
gt = {}
pred = {}
##
judge_detections(gt, pred)