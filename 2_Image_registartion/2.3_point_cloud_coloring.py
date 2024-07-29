# This code is for point cloud coloring.

import cv2
import numpy as np
import open3d as o3d

image1 = cv2.imread("D:\\Crack_quantification\\results\\3.1_output_intensity_image.jpg")
image2 = cv2.imread("D:\\Crack_quantification\\results\\3.2_ouput_optical_image_after_transformation.jpg")
image2 = cv2.resize(image2,(image1.shape[1],image1.shape[0]))

gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray_image1, 240, 255, cv2.THRESH_BINARY)
image1[mask != 255] = image2[mask != 255]
cv2.imwrite("D:\\Crack_quantification\\results\\2.3_ouput_colored_intensity_image.jpg", image1)
image3 = cv2.imread("D:\\Crack_quantification\\results\\2.3_ouput_colored_intensity_image.jpg")

xoy_path = "D:\Crack_quantification\\results\\1.1_ouput_pier_xoy.pcd"
inside_path = "D:\Crack_quantification\\results\\1.1_ouput_pier_inside.pcd"
pcd = o3d.io.read_point_cloud(xoy_path)
pcd3 = o3d.io.read_point_cloud(inside_path)
pcd2= np.asarray(pcd.points)
pcd4= np.asarray(pcd3.points)
x_min, y_min, z_min = np.amin(pcd2, axis=0)
x_max, y_max, z_max = np.amax(pcd2, axis=0)
points_x = np.asarray(pcd2[:, 0])
points_y = np.asarray(pcd2[:, 1])
points_z = np.asarray(pcd2[:, 2])
points_xx = np.asarray(pcd4[:, 0])
points_yy = np.asarray(pcd4[:, 1])
points_zz = np.asarray(pcd4[:, 2])

def divide_points(x, y, a, b, n, c, d, m):
    x_interval = (b - a) / n
    y_interval = (d - c) / m
    x_indices = np.floor((x - a) / x_interval).astype(int)
    y_indices = np.floor((y - c) / y_interval).astype(int)
    x_indices = np.clip(x_indices, 0, n - 1)
    y_indices = np.clip(y_indices, 0, m - 1)
    return list(zip(x_indices, y_indices))

def write_2d_array_to_empty_file(file_path, array_values):
    with open(file_path, 'w') as file:
        for row in array_values:
            row_str = ' '.join(str(value) for value in row)
            file.write(row_str + '\n')

groups = divide_points(points_x, points_y, x_min, x_max, image1.shape[0]-20, y_min, y_max, image1.shape[1]-20)
tb=[]
pcd1 = o3d.geometry.PointCloud()
for i in range(len(pcd2)):
    ss=image2[10+groups[i][0]][image1.shape[1]-10-groups[i][1]]
    tb.append([points_xx[i],points_yy[i],points_zz[i],ss[2],ss[1],ss[0]])
file_path = 'D:\Crack_quantification\\results\\2.3_ouput_colored_point_cloud.txt'
xx= np.ones((image1.shape[0],image1.shape[1],3),np.uint8)*255
for i in range(len(tb)):
    xx[groups[i][0]][groups[i][1]]=[tb[i][-3],tb[i][-2],tb[i][-1]]
cv2.imshow("colored_point_cloud",xx)
cv2.waitKey(0)
write_2d_array_to_empty_file(file_path, tb)
