# This code is for line segments enhancement of intensity images.

import cv2
import numpy as np
import math

# Draw lines on the image.
def draw_line(point1, point2, image):
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    for x in range(min(point1[0], point2[0]), max(point1[0], point2[0])):
        y = int(slope * (x - point1[0]) + point1[1])
        avg_color = np.mean(image[y-1:y+2, x:x+2], axis=(0, 1))
        image[y, x] = avg_color.astype(int)
# Calculate angles.
def calculate_angle(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    angle = math.degrees(math.atan(slope))
    return angle

image = cv2.imread('D:\Crack_quantification\\results\\1.1_output_intensity_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
line_lengths = [(line, np.linalg.norm((line[0][0] - line[0][2], line[0][1] - line[0][3]))) for line in lines]
line_lengths.sort(key=lambda x: x[1], reverse=True)
longest_lines = line_lengths[:40]

for line, _ in longest_lines:
    x1, y1, x2, y2 = line[0]
    if abs(calculate_angle(x1, y1, x2, y2)) > 60 or abs(calculate_angle(x1, y1, x2, y2)) < 0 :
        color1 = image[y1, x1] 
        color2 = image[y2, x2]
        average_color = np.mean([color1, color2], axis=0)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 4)

cv2.imshow('Longest Five Lines', image)
cv2.imwrite("D:\Crack_quantification\\results\\2.1_output_intensity_image_after_line_enhancement.jpg",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
