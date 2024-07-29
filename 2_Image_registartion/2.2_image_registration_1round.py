# This code is for image registration.

import sys
sys.path.append("D:\Crack_quantification")
from copy import deepcopy
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from LoFTR.src.utils.plotting import make_matching_figure
from LoFTR.src.loftr import LoFTR, default_cfg

size = (448, 448) # Resize the images to a uniform size.
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("D:\\Crack_quantification\\LoFTR\\demo\\outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()
default_cfg['coarse']

img0_pth = r"D:\\Crack_quantification\\results\\2.1_ouput_intensity_image_after_line_enhancement.png"
img1_pth = r"D:\\Crack_quantification\\results\\3.1_output_optical_image_after_stitching.png"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, size)
img1_raw = cv2.resize(img1_raw, size)
img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

# Inference with LoFTR and get prediction
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
mconf_fila = []
m0 =[]
m1=[]
for i in range(len(mconf)):
    if mconf[i] >0.30:
        if np.abs(mkpts0[i][0]-mkpts1[i][0]) < 180 and np.abs(mkpts0[i][1]-mkpts1[i][1]) < 180:
            mconf_fila.append(mconf[i])
            m0.append(mkpts0[i])
            m1.append(mkpts1[i])
m0 = np.array(m0)
m1 = np.array(m1)
print(len(mconf_fila))
color = cm.jet(mconf_fila)
text = [
    'LoFTR',
    'Matches: {}'.format(len(m0)),
]
fig = make_matching_figure(img0_raw, img1_raw, m0, m1, color, text=text, path="D:\\Crack_quantification\\results\\image_registration.png")
imb = cv2.imread(img0_pth)
ima = cv2.imread(img1_pth)
A = ima.copy()
B = imb.copy()
imageA = cv2.resize(A,size)
imageB = cv2.resize(B,size)
H, status = cv2.findHomography(m1, m0, cv2.USAC_MAGSAC, 0.8, 1, 100000)
result1 = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0]))
cv2.imwrite("D:\Crack_quantification\\results\\3.2_output_optical_image_after_transformation.jpg", result1)
