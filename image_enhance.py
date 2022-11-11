import cv2
import numpy as np
from tqdm import tqdm
import os
import glob

def cla_he(img):
  img2=img.copy()
  clahe=cv2.createCLAHE(clipLimit = 5, tileGridSize = (14,14))
  for i in range(3):
    img2[:,:,i]=clahe.apply(img[:,:,i])
  return img2


def gama_balance(image):
    img = image.copy()
    img = img / 255.0
    for i in range(3):
        img[:, :, i] = np.power(img[:, :, i] / float(np.max(img[:, :, i])), 2)
    img = np.clip(img * 255, 0, 255)
    img = np.uint8(img)
    return img

input_file='train_images'
output_file='output'

names=glob.glob(input_file+'/*/*.jpg')
if not os.path.exists(output_file):
  os.mkdir(output_file)
if not os.path.exists(os.path.join(output_file,'video_0')):
  os.mkdir(os.path.join(output_file,'video_0'))
if not os.path.exists(os.path.join(output_file,'video_1')):
  os.mkdir(os.path.join(output_file,'video_1'))
if not os.path.exists(os.path.join(output_file,'video_2')):
  os.mkdir(os.path.join(output_file,'video_2'))
for name in tqdm(names):
  img=cv2.imread(name)
  img=gama_balance(cla_he(img))
  cv2.imwrite(output_file+name[len(input_file):],img)
