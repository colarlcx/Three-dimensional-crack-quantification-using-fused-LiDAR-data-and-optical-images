# After 1round, we got optical image after perspective transformation.
# But, its resolution is 448*448, which is insufficient for the crack quantification.
# Thus, this code is to obtain an optical image from the same viewpoint as the point cloud intensity map, ensuring no loss of precision.

import sys
sys.path.append("D:\Crack_quantification")
import cv2
import numpy as np
import sys
from LoFTR.src.utils.plotting import make_matching_figure
import matplotlib.cm as cm
import random

# Detect SIFT key feature points in images A and B, and calculate the feature descriptors.
def detectAndDescribe(image):
    sift = cv2.SIFT_create()
    (kps, features) = sift.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)

size = (4480,8400)  # The orginal size of optical pictures taken by iphone.
img1_pth = "D:\\Crack_quantification\\results\\3.1_ouput_optical_image_after_stitching.png"
img0_pth = "D:\\Crack_quantification\\results\\3.2_ouput_optical_image_after_transformation.jpg"
img_mask_pth = "D:\\Crack_quantification\\raw_data\\crack_mask.png"
ima = cv2.imread(img1_pth)
imb = cv2.imread(img0_pth)
imb = cv2.resize(imb, size)
A = ima.copy()
B = imb.copy()
imageA = cv2.resize(A, size)
imageB = cv2.resize(B, size)
imageC= cv2.imread("D:\PythonFiles\PytorchFiles\YYGoFighting\pc/labelme\labelme23large_mask_jiaozheng.png")
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw,  size)
img1_raw = cv2.resize(img1_raw,  size)


kpsA, featuresA = detectAndDescribe(imageA)
kpsB, featuresB = detectAndDescribe(imageB)

bf = cv2.BFMatcher()
random_values_A = random.sample(range(1, len(featuresA)), 40000)
random_values_B = random.sample(range(1, len(featuresB)), 4000)
kpsAA=[]
kpsBB=[]
featuresAA=[]
featuresBB=[]
for i in random_values_A:
    kpsAA.append(kpsA[i])
    featuresAA.append(featuresA[i])
for i in random_values_B:
    kpsBB.append(kpsB[i])
    featuresBB.append(featuresB[i])
featuresAA=np.array(featuresAA).astype(np.float32)
featuresBB=np.array(featuresBB).astype(np.float32)

# Use K-Nearest Neighbors (KNN) to detect SIFT feature matches between images A and B, with K=2.
matches = bf.knnMatch(featuresAA, featuresB, 2)
good = []
mconf_fila = []
for m in matches:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.7:
        good.append((m[0].trainIdx, m[0].queryIdx))
        mconf_fila.append(1-(m[0].distance/m[1].distance))

kk=0
goods=[]
sb=[]
for i in range(len(good)):
    ia = good[i][1]
    ib = good[i][0]
    if np.abs(kpsAA[ia][0]-kpsB[ib][0]) < 1800 and np.abs(kpsAA[ia][1]-kpsB[ib][1]) < 1800:
        goods.append(good[i])
        sb.append(mconf_fila[i])
    else:
        kk =kk+1

# When the filtered matching pairs exceed 4, calculate the homography matrix for perspective transformation.
if len(good) > 4:
    ptsA = np.float32([kpsAA[i] for (_, i) in goods])
    ptsB = np.float32([kpsB[i] for (i, _) in goods])
    H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,2.0)
M = (matches, H, status)
if M is None:
    sys.exit()
(matches, H, status) = M

result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
result1 = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0]))
result2 = cv2.warpPerspective(imageC, H, (imageA.shape[1], imageA.shape[0]))
cv2.imwrite("D:\Crack_quantification\\results\\3.2_ouput_optical_image_after_transformation_high_resolution.png",result1)
cv2.imwrite("D:\Crack_quantification\\results\\3.2_output_crack_mask_after_transformation.png",result2)