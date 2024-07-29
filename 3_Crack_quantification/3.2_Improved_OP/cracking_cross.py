import cv2
import numpy as np
import math

# Calculate the degree of a point (xx, yy) in the skeleton.
def cal_len(xx,yy,it,flag=True):
    k=0
    points = []
    for itt in it:
        x = itt[0]
        y = itt[1]
        if (xx-x)**2 + (yy-y)**2 <= 2:
            k = k+1
            points.append((x, y))
        else:
            pass
    if flag:
        return k
    else:
        return points

# Starting from a set of intersection points, find the nearest skeleton point, 
# then find the nearest skeleton point again, 
# and repeat the process (for points within a distance of less than 2).
def find_single_crack(given_point,points):
    distances = np.linalg.norm(points - given_point, axis=1)
    nearest_point_index = np.where(distances==0)
    points = np.delete(points, nearest_point_index, axis=0)
    distances = np.linalg.norm(points - given_point, axis=1)
    nearest_point_index = np.where(distances<2)
    co = points[nearest_point_index]
    points = np.delete(points, nearest_point_index, axis=0)
    return co,points

# Compute the normal direction at each skeleton point.
def calculate_moments(coordinates):
    points = np.array(coordinates)
    x_mean = np.mean(points[:, 0])
    y_mean = np.mean(points[:, 1])
    mu_xx = np.mean((points[:, 0] - x_mean)**2)+1/12
    mu_yy = np.mean((points[:, 1] - y_mean)**2)+1/12
    mu_xy = np.mean((points[:, 0] - x_mean) * (points[:, 1] - y_mean))
    if (not mu_xy):
        if all(x == x_mean for x in points[:,0]):
            theta = math.pi
        elif all(y == y_mean for y in points[:,0]):
            theta = math.pi/2
        else:
            theta = math.pi
    else:
        theta = math.atan((abs(mu_xx - mu_yy) + math.sqrt(abs((mu_xx - mu_yy) ** 2 - 4 * mu_xy ** 2))) / (2 * mu_xy))
    normal_direction = theta + math.pi/2 
    return theta, normal_direction

# Calculate the tangent directions and normal directions of the local skeleton points.
def cal_cracking_skeleton_orientation_normal(points_list, end_point_flag = False, n=0):
    normals=[]
    if end_point_flag:
        for i in range(len(points_list)):
            if i in {0,-1}:
                zz=5+n
            else: 
                zz=3+n
            close_points = [point for point in points_list if abs(point[0] - points_list[i][0]) < zz and abs(point[1] - points_list[i][1]) < zz]
            normals.append(calculate_moments(close_points)[1])
    else:
        for i in {0,-1}:
            zz=5
            close_points = [point for point in points_list if abs(point[0] - points_list[i][0]) < zz and abs(point[1] - points_list[i][1]) < zz]
            print(close_points)
            normals.append(calculate_moments(close_points)[1])
    return normals

# Draw the tangent directions of the skeleton points on the given image.
def draw_cracking_skeleton_orientation_normal(ii,cracks, picture=None):
    centers = np.array(cracks).reshape(-1, 2)[ii]
    center = tuple(x - 50 for x in centers)
    angle_degrees = 360*cal_cracking_skeleton_orientation_normal(np.array(cracks).reshape(-1, 2))[ii]/2/math.pi
    angle_radians = math.radians(angle_degrees)
    line_length = 10 
    begin_point = (int(center[0] - line_length * math.cos(angle_radians)), int(center[1] - line_length * math.sin(angle_radians)))
    end_point = (int(center[0] + line_length * math.cos(angle_radians)), int(center[1] + line_length * math.sin(angle_radians)))
    color = (0, 0, 0)
    thickness = 2
    if len(picture):
        cv2.line(picture, begin_point, end_point, color, thickness)
    return (begin_point,end_point)

# To find cracks or contours in a given image.
def find_contours(picture,draw_flag=False):
    gray_image = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    contour_points = []
    for contour in contours:
        for point in contour:
            contour_points.append((point[0][0],point[0][1]))
    result_image = np.ones((len(gray_image),len(gray_image[0]),3)).astype(np.uint8)*255
    if draw_flag:
        for item in contour_points:
            result_image[item[1],item[0]] = (0,0,0)
        # cv2.imshow("re",result_image)
    else:
        pass
    return contour_points, result_image.astype(np.uint8)