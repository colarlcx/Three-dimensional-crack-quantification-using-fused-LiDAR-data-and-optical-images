# This code is for pixel resolution calculation.
import cv2
import numpy as np
image = cv2.imread("D:\Crack_quantification\\results\\3.2_ouput_colored_intensity_image.jpg")
image = cv2.resize(image, (4480, 8400))
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
internal_contours = []
for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        internal_contours.append(contours[i])
contour_image = np.copy(image)
cv2.drawContours(contour_image, internal_contours, -1, (0, 255, 0), 3)
internal_contour_area = 0
for contour in internal_contours:
    internal_contour_area += cv2.contourArea(contour)
s_point_cloud = 18.30885 # This is obtained by cloudcompare.
pr = internal_contour_area/1000000/s_point_cloud # Pixel resolution.