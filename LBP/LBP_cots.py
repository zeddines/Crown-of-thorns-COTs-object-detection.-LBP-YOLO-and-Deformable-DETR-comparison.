from skimage import feature
from skimage import feature
import os
import numpy as np
import cv2 as cv
import pandas as pd
from skimage.color import label2rgb
import matplotlib.pyplot as plt 
import copy
import json
import re
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
        
    def describe(self, image, box, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        lbp = np.array(lbp)
        print(lbp)
        lbp = lbp[box["ymin"]:(box["ymin"]+box["height"]), box["xmin"]:(box["xmin"]+box["width"])]
        print(lbp)
        (hist, edges) = np.histogram(lbp.ravel(),
		    bins=np.arange(0, self.numPoints + 3),
		    range=(0, self.numPoints + 2))
		# normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        fig, ax = plt.subplots()
        ax.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black", align="edge")
        plt.show()
		# return the histogram of Local Binary Patterns
        #return hist
    
    def display(self, image, GT, box, img2):
        
        img2 = copy.deepcopy(img2)
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        print(np.min(lbp))
        print(np.max(lbp))
        lbp = np.uint8(lbp)
        lbp_display = contrast_stretching(lbp, 0, 255)
        unique, counts = np.unique(lbp, return_counts=True)
        lbp = add_bb(lbp, GT)
        print(dict(zip(unique, counts)))
        #img = np.zeros((lbp.shape[0], lbp.shape[1], 3))
        colors = [tuple(np.random.randint(0, 256, size=3)) for i in range(0, len(np.unique(lbp)))]
        img = label2rgb(lbp, colors = colors)
        cv.rectangle(img2, (box["xmin"], box["ymin"]), (box["xmin"]+box["width"], box["ymin"]+box["height"]), (0, 255, 0), thickness = 2)
        cv.rectangle(lbp_display, (box["xmin"], box["ymin"]), (box["xmin"]+box["width"], box["ymin"]+box["height"]), (255, 255, 255), thickness = 2)
        add_bb(img2, GT)
        """
        for c, i in enumerate(np.unique(lbp)):
            indices = np.where(lbp == i)
            indices = tuple(zip(*indices))
            for j in indices:
                img[j[0], j[1], 0] = colors[c][0]
                img[j[0], j[1], 1] = colors[c][1]
                img[j[0], j[1], 2] = colors[c][2]
        cv.imwrite("lbpmap2.jpg", img)
        """
        cv.imwrite("lbpmap.jpg", img)
        cv.imwrite("lbp.jpg", lbp_display)
        cv.imwrite("img_wbox.jpg",img2)

def contrast_stretching(img, l, u):
    minV, maxV = np.min(img), np.max(img)
    print(minV)
    print(maxV)
    newImg = np.empty(img.shape)
    for r in range(0, img.shape[0]):
        for c in range(0, img.shape[1]):
            newImg[r, c] = (img[r, c]-minV)*((u-l)/(maxV-minV))+l
    return newImg


def add_bb(img, GT):
    color = (255, 0, 0)
    for b in GT:
        img = cv.rectangle(img, (b["x"], b["y"]), (b["x"]+b["width"], b["y"]+b["height"]), color, thickness = 2)
    return img

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

#params
   
#img_path = os.path.join("video_0", "9375.jpg")    
#img_id = "0-9375"
#box =  {"xmin":809, "ymin":424, "width":70, "height":56} #fake
#box =  {"xmin":630, "ymin":554, "width":77, "height":70} #real
#box =  {"xmin":601, "ymin":261, "width":76, "height":52} #real

img_path = os.path.join("video_0", "1017.jpg") 
img_id = "0-1017"
P = 8
R = 0.5
box = {'xmin': 1045, 'ymin': 446, 'width': 98, 'height': 67} #real
###

df = pd.read_csv(os.path.join("tensorflow-great-barrier-reef", 'train.csv'), converters={'annotations': lambda x: convert_annotaions_to_dicts(x)})
_, ann_w_class = split_annotate_with_without_class(df)

img = cv.imread(os.path.join("tensorflow-great-barrier-reef", "train_images", img_path), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(os.path.join("tensorflow-great-barrier-reef", "train_images", img_path), cv.IMREAD_COLOR)
lbp = LocalBinaryPatterns(P, R)
lbp.display(img, ann_w_class[img_id], box, img2)
lbp.describe(img, box)

"""
grad_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
grad_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)

grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
grad = add_bb(grad, ann_w_class[img_id])
cv.imwrite("sobel.jpg", grad)
cv.imshow("original",add_bb(img, ann_w_class[img_id]))
cv.imshow("sobel", grad)
cv.waitKey(0)
"""

