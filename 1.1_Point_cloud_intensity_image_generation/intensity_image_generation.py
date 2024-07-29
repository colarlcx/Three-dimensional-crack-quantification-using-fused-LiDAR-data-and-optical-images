# This code is for intensity image generation.
import numpy as np
import open3d as o3d
import cv2

def plane_tran_pcd(pcd, plane_normal, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(plane_normal * angle_radians)
    pcd.rotate(R, center=pcd.get_center())
    pcdp= np.asarray(pcd.points)
    x_min, y_min, z_min = np.amin(pcdp, axis=0)
    x_max, y_max, z_max = np.amax(pcdp, axis=0)
    return ((x_max-x_min)*(y_max-y_min))

def read_and_convert_to_open3d(input_file):
    data = np.loadtxt(input_file)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data[:, :3])
    point_cloud.colors = o3d.utility.Vector3dVector(data[:, 3:] / 255.0) 
    return point_cloud

# Use the Rodrigues formula to rotate a set of points between two normal vectors.
def rodrigues_rot(P, n0, n1):
    P = np.atleast_2d(P)
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    k = np.cross(n0, n1)
    P_rot = np.zeros((len(P), 3))
    if np.linalg.norm(k) != 0:
        k = k / np.linalg.norm(k)
        theta = np.arccos(np.dot(n0, n1))
        # Compute rotated points
        for i in range(len(P)):
            P_rot[i] = (
                P[i] * np.cos(theta) + np.cross(k, P[i]) * np.sin(theta) + k * np.dot(k, P[i]) * (1 - np.cos(theta))
            )
    else:
        P_rot = P
    return P_rot

# Project the point cloud onto the specified plane.
def point_cloud_plane_project(cloud, coefficients):
    A = coefficients[0]
    B = coefficients[1]
    C = coefficients[2]
    D = coefficients[3]
    Xcoff = np.array([B * B + C * C, -A * B, -A * C])
    Ycoff = np.array([-B * A, A * A + C * C, -B * C])
    Zcoff = np.array([-A * C, -B * C, A * A + B * B])
    points = np.asarray(cloud.points)
    xp = np.dot(points, Xcoff) - A * D
    yp = np.dot(points, Ycoff) - B * D
    zp = np.dot(points, Zcoff) - C * D
    project_points = np.c_[xp, yp, zp]
    project_cloud = o3d.geometry.PointCloud() 
    project_cloud.points = o3d.utility.Vector3dVector(project_points)
    project_cloud.colors = cloud.colors
    return project_cloud

# Straighten the image and crop the middle section.
def crop(image_path="D:\\Crack_quantification\\results\\intensity_image.png", n=10):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    points = np.column_stack(np.where(threshold == 0))
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    
    x, y, x2, y2 = min_x-n, min_y-n, max_x+n, max_y+n
    cropped_image = image[x:x2,y:y2]
    if x2-x-y2+y < 0:
        cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite('D:\\Crack_quantification\\results\\intensity_image.png', cropped_image)

if __name__ == '__main__':
    input_file = "D:\\Crack_quantification\\raw_data\\original_point_cloud.txt"
    pcd = read_and_convert_to_open3d(input_file)

    # RANSAC plane fitting.
    plane_model, inliers = pcd.segment_plane(distance_threshold=1,
                                    ransac_n=3,
                                    num_iterations=1000000)
    [a, b, c, d] = plane_model
    inlier_cloud = pcd.select_by_index(inliers)
    o3d.visualization.draw_geometries([inlier_cloud], window_name=f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0",
                                    width=900, height=900,
                                    left=50, top=50,
                                    mesh_show_back_face=False)
    o3d.io.write_point_cloud("D:\\Crack_quantification\\results\\second_fanwei_pier_neidian2.pcd", inlier_cloud)
    all_liers = index = np.arange(0,len(pcd.points),1).astype(np.int64)
    outliers = np.setdiff1d(all_liers,inliers)
    outlier_cloud = pcd.select_by_index(outliers)
    projected_cloud = point_cloud_plane_project(inlier_cloud, plane_model) # Project the point cloud onto the specified plane.
    transp = rodrigues_rot(projected_cloud.points,np.array([a,b,c]),np.array([0,0,1])) #  Project the point cloud onto the XOY plane.
    b = np.asarray(transp, dtype = np.float16)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(b)
    pcd1.colors = projected_cloud.colors
    pcd2= np.asarray(pcd1.points)
    x_min, y_min, z_min = np.amin(pcd2, axis=0)
    x_max, y_max, z_max = np.amax(pcd2, axis=0)
    # Align the point cloud to the XY plane.
    min = (x_max-x_min)*(y_max-y_min)
    index = 0
    for i in range(360):
        mianji = plane_tran_pcd(pcd1, np.array([0,0,1]), 1)
        if  mianji < min:
            index = i
            min = mianji
    plane_tran_pcd(pcd1, np.array([0,0,1]), 1*index+1)
    o3d.visualization.draw_geometries([pcd1], 
                                    width=900, height=900,
                                    left=50, top=50,
                                    mesh_show_back_face=False)
    o3d.io.write_point_cloud("D:\\Crack_quantification\\results\\second_fanwei_pier_xoy2.pcd", pcd1)
    
    # Get the inetnsity image by o3d library.
    pcd1.scale(1.0, center=pcd1.get_center())
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920,height=1080)
    vis.add_geometry(pcd1)
    vis.update_geometry(pcd1)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("D:\\Crack_quantification\\results\\intensity_image.png")
    vis.destroy_window()
    crop()
