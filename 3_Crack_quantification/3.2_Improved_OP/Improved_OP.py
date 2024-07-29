# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage.morphology import medial_axis
from skimage.util import invert
from cracking_cross import cal_len, find_single_crack, find_contours, cal_cracking_skeleton_orientation_normal
from sklearn.cluster import DBSCAN
import math
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import math
#---------------------------DCE:------------------------------#
def find_white_point_number(picture0,picture1):
    gray_image = np.uint8(picture0)
    gray_image1 = np.uint8(picture1)
    intersection_points = np.argwhere((np.logical_and(gray_image == 255, gray_image1==255)))
    result = intersection_points[:, :2]
    unique_result = np.unique(result, axis=0)
    return len(unique_result)

def find_cross_point(start_point,end_point,picture):
    gray_image = np.ones_like(picture)*255
    cv2.line(gray_image, start_point, end_point, 0, 1)
    gray_image = np.uint8(gray_image)
    intersection_points = np.argwhere(gray_image == 0)
    result = intersection_points[:, :2]
    unique_result = np.unique(result, axis=0)
    return unique_result

def farthest_points(points):
    max_distance = 0
    farthest_pair = ()
    for pair in itertools.combinations(points, 2):
        distance = math.sqrt((pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2)
        if distance > max_distance:
            max_distance = distance
            farthest_pair = pair
    return farthest_pair

def draw_cracking_skeleton_orientation_normal(ii,cracks,t, picture,jj=0):
    centers = cracks[ii]
    center = (centers[1], centers[0])
    angle_degrees = 360*t/2/math.pi
    angle_radians = math.radians(angle_degrees)
    line_length = 15
    begin_point = (int(center[0] - line_length * math.cos(angle_radians)), int(center[1] - line_length * math.sin(angle_radians)))
    end_point = (int(center[0] + line_length * math.cos(angle_radians)), int(center[1] + line_length * math.sin(angle_radians)))
    color = (255, 255,255)
    thickness = 2 
    if jj:
        cv2.line(picture, begin_point, end_point, color, thickness)
    return (begin_point,end_point)

def group_points_by_op(points,given_vector, ps, jj=0):
    direction_vectors = points - np.array([ps[0],ps[1]])
    if jj==1: # OP Method
        dot_products=[]
        for i in range(len(direction_vectors)):
            dot_products.append(np.dot(direction_vectors[i], given_vector)/np.linalg.norm(direction_vectors[i]))
        dot_products = np.array(dot_products)
        min_index = np.argmin(dot_products)
        max_index = np.argmax(dot_products)
        return points[min_index],points[max_index]
    elif jj==2:
        points = np.array(points)
        dot_products=[]
        for i in range(len(direction_vectors)):
            dot_products.append(np.dot(direction_vectors[i], given_vector)/np.linalg.norm(direction_vectors[i]))
        dot_products = np.array(dot_products)
        d1 = points[np.where(dot_products>0.7)]
        d2 = points[np.where(dot_products<-0.7)]
        juli = np.linalg.norm(d1[0]-d2[0])
        index=[0,0]
        for i in range(len(d1)):
            for j in range(len(d2)):
                if np.linalg.norm(d1[i]-d2[j])<juli:
                    juli = np.linalg.norm(d1[i]-d2[j])
                    index[0]=i
                    index[1]=j
        return d1[index[0]],d2[index[1]]
    else:
        dot_products =[]
        points = np.array(points)
        for i in range(len(direction_vectors)):
            dot_products.append(np.dot(direction_vectors[i], given_vector)/np.linalg.norm(direction_vectors[i]))
        dot_products = np.array(dot_products)
        p1 = points[np.where(dot_products>0)]
        p2 = points[np.where(dot_products<0)]
        d1=np.linalg.norm(np.array(p1)- np.array([ps[0],ps[1]]), axis=1)
        d2=np.linalg.norm(np.array(p2)- np.array([ps[0],ps[1]]), axis=1)
        c1 = np.argmin(d1)
        c1p= p1[c1]
        c2 = np.argmin(d2)
        c2p= p2[c2]
        if jj ==0:
            return c1p, c2p
        else:
            return p1,p2
def OrthoBoundary(p1,p2):
    return theta

# Crack_quantification
def COT(raw_image, func="D", das ="C"):
    def dfs(G, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        paths = [[start]]
        for neighbor in G.neighbors(start):
            if neighbor not in visited:
                for path in dfs(G, neighbor, visited):
                    paths.append([start] + path)
        return paths
    # DCE
    def huatu(skee, skep, ra, delete_p,i):
        ske = np.zeros_like(skee)
        for k in range(len(skeleton_points)):
            if (skep[k][1],skep[k][0]) not in delete_p:
                cv2.circle(ske, (skep[k][1],skep[k][0]), 1,  255, int(ra[k]))
        return ske
    # C2T
    def cal_crack_length(cp):
        l = 0
        for i in range(len(cp)-1,0,-1):
            item = cp[i][0]
            item1 = cp[i-1][0]
            if np.abs(item1[0]-item[0])>np.abs(item1[1]-item[1]):
                l =np.abs(item1[0]-item[0])+(math.sqrt(2)-1)*np.abs(item1[1]-item[1])+l
            else:
                l =(math.sqrt(2)-1)*np.abs(item1[0]-item[0])+np.abs(item1[1]-item[1])+l
        return l
    if das == "C":
        image=cv2.imread(raw_image,0)
        rgb_image=np.array(cv2.imread(raw_image))
    else:
        image=cv2.imread(raw_image,0)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # cv2.imshow("yuantu", rgb_image)
    inverted_image = invert(image)
    _, binary = cv2.threshold(inverted_image, 128, 255, cv2.THRESH_BINARY_INV)
    binary[binary == 255] = 1
    skeleton0, dis = medial_axis(binary, return_distance=True)
    skeleton = skeleton0.astype(np.uint8)*255
    skeleton_points = np.argwhere((skeleton == 255))#
    if func!="C":
        pic1 = binary.astype(np.uint8)*255
        indices = np.where(skeleton == 255)
        indices = np.transpose(indices)
        indi = [tuple(item) for item in indices]
        rad =[]
        for i in range(len(image)):
            for j in range(len(image[0])):
                if (i,j) in indi:
                    rad.append(dis[i][j])
        total_white_point_number=find_white_point_number(pic1,huatu(skeleton, skeleton_points, rad,[],10))
        pass
    nod = [] 
    sas = []
    sbs = []
    endpoint = [] 
    beginpoint = []
    for item in skeleton_points:
        x = item[0]
        y = item[1]
        flag = cal_len(x,y,skeleton_points)
        if flag>3:
            sbs.append((y,x))
        elif flag==2:
            nod.append(np.array([[y,x]]))
            endpoint.append((y,x))
            sas.append((y,x))
        else:
            sas.append((y,x))
            pass
    aaa = [list(t) for t in sas]
    assb =np.array(aaa)
    ssa = np.array([list(t) for t in sbs])
    p=[]
    dbscan = DBSCAN(eps=math.sqrt(2), min_samples=1)
    clusters = dbscan.fit_predict(np.array(ssa))
    for i in range(len(set(clusters))):
        nod.append(ssa[clusters==i])
        p.append(ssa[clusters==i])
    kan =[]
    for item in sbs:
        if cal_len(item[0],item[1],sas):
            beginpoint.append((item[0], item[1]))
        else:
            kan.append(item)
    p1 =[]
    for kk in range(len(ssa)):
        kx=assb[np.linalg.norm(assb - np.array(ssa[kk]), axis=1)<2]
        for kxx in kx:
            p1.append(kxx)
    sp = []
    p3=[]
    for item in p1:
        if cal_len(item[0],item[1],ssa,True)!=1:
            k1,k2=cal_len(item[0],item[1],ssa,False)
            p3.append([[[k1[0],k1[1]]],[[item[0],item[1]]],[[k2[0],k2[1]]],[0,0]])
        else:
            sp.append(item)    
    index =[]
    for i in range(len(sp)): 
        bp = cal_len(sp[i][0],sp[i][1],ssa,False)[0]
        p2=[]
        p2.append([[bp[0],bp[1]]])
        p2.append([[sp[i][0],sp[i][1]]])
        x,pp=find_single_crack(np.array(sp[i]),np.array(assb))
        while len(x):
            p2.append(x)
            x,pp=find_single_crack(x[0],pp)
        xx= p2[-1]
        if (xx[0][0],xx[0][1]) not in endpoint:
            ep=cal_len(xx[0][0],xx[0][1],ssa,False)[0]
            p2.append([[ep[0],ep[1]]])
            p2.append([0,0])
        else:
            p2.append([1,1])
        flag =True
        if len(p2)!=0:
            p2j = np.array(p2[:-1])
            for item in p3:
                item = np.array(item)
                if np.abs(len(item) - len(p2j))<2 and np.linalg.norm(p2j[0][0]-item[-1][0])+ np.linalg.norm(p2j[-1][0]-item[0][0])<=4:
                    flag = False
                else:
                    pass
            if flag:
                p3.append(p2)
        else:
            pass
    single_components = np.ones_like(rgb_image, dtype=np.uint8)*255
    colors = [np.random.randint(0, 255, 3) for _ in range(len(p3))]
    if func=="C":
        for i in range(len(p3)):
            for item in p3[i][:-1]:
                single_components[int(item[0][1])][int(item[0][0])] = colors[i]
        for item in kan:
            single_components[int(item[1])][int(item[0])] = (0,0,0)
    else:
        for i in range(len(p3)):
            if len(p3[i]):# sbbb[i]>0
                if p3[i][-1][0]==0:
                    for item in p3[i][:-1]:
                        single_components[int(item[0][1])][int(item[0][0])]=(0,0,0)  
                else:
                    dp = [(int(item[0][0]),int(item[0][1])) for item in p3[i][:-1]] 
                    pic2 = huatu(skeleton, skeleton_points, rad, dp,i)
                    local_wpn=find_white_point_number(pic1, pic2)
                    if total_white_point_number-local_wpn>75:
                        for item in p3[i][:-1]:
                            single_components[int(item[0][1])][int(item[0][0])]=(0,0,0)  
                    else:
                        for item in p3[i][:-1]:
                            single_components[int(item[0][1])][int(item[0][0])]=(0,255,0)  
    # cv2.imshow("jieguo",single_components)
    for item in sas:
        single_components[item[1]][item[0]]=(0,255,0)
    for item in sbs:
        single_components[item[1]][item[0]]=(0,0,255)
    colors = [np.random.randint(0, 255, 3) for _ in range(len(clusters))]
    for i in range(len(p)):
        for item in p[i]:
            single_components[item[1]][item[0]]=colors[i]
    # cv2.imshow('Dilated', single_components)
    # Create an graph.
    G = nx.Graph()
    G.add_node(1)
    G.add_nodes_from([i+2 for i in range(len(nod)-2)])
    nod = np.array(nod,dtype=object)
    for i in range(len(p3)):
        ei=-1
        bi =-1
        for j in range(len(nod)):
            for k in range(len(nod[j])):
                if p3[i][0][0][0] == nod[j][k][0] and p3[i][0][0][1] == nod[j][k][1]:
                    bi = j
                if p3[i][-2][0][0] == nod[j][k][0] and p3[i][-2][0][1] == nod[j][k][1]:
                    ei = j
        if (bi,ei) not in index and (ei,bi) not in index:
            G.add_edge(bi,ei,length=cal_crack_length(p3[i][:-1]))
        else:
            pass
        index.append((bi,ei))
    path_lengths = nx.single_source_dijkstra_path_length(G, 1, weight='length')

    # Find the farest node.
    farthest_node = max(path_lengths, key=path_lengths.get)
    path = nx.single_source_dijkstra_path(G, farthest_node)
    keys = list(path.keys())
    s2 = keys[0]
    pn0 = path[s2]
    ki = sum([G[pn0[i]][pn0[i+1]]['length'] for i in range(len(pn0)-1)])
    for ii in keys:
        pn = path[ii]
        lpl = sum([G[pn[i]][pn[i+1]]['length'] for i in range(len(pn)-1)])
        if lpl > ki:
            ki = lpl
            s2 =ii
    longest_path_nodes = path[s2]
    nx.draw(G, with_labels=True)
    single_componentss = np.ones_like(rgb_image, dtype=np.uint8)*255
    results=[]
    for i in range(len(p3)):
        if index[i][0] in longest_path_nodes and index[i][1] in longest_path_nodes:
                for item in p3[i][:-1]:
                    single_componentss[int(item[0][1])][int(item[0][0])]=(0,0,0)
                    results.append((int(item[0][1]),int(item[0][0])))  
    for item in kan:
        single_componentss[int(item[1])][int(item[0])] = (0,0,0)
        results.append((int(item[1]),int(item[0]))) 
    return results
def check12p(center_point, direction_vector):
    neighbor_points = np.array([
        [center_point[0]-1, center_point[1]-1], [center_point[0]-1, center_point[1]], [center_point[0]-1, center_point[1]+1],
        [center_point[0], center_point[1]-1],[center_point[0], center_point[1]+1],
        [center_point[0]+1, center_point[1]-1], [center_point[0]+1, center_point[1]], [center_point[0]+1, center_point[1]+1]
    ])
    vectors = neighbor_points - center_point
    dot_products = np.dot(vectors, direction_vector)
    angles = np.arccos(dot_products / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(direction_vector)))
    tolerance = 0.1 
    matching_points = neighbor_points[(angles < tolerance) | (angles > np.pi - tolerance)]
    return matching_points

if __name__=="__main__":
    OB = 0
    raw_image = r"D:\\yuan5.jpg"
    sp=COT(raw_image,"D","C")
    contours, con_img = find_contours(np.array(cv2.imread(raw_image)),True)
    for item in sp:
        con_img[item[0]][item[1]] = (0,0,255)
    pic =np.array(cv2.imread(raw_image))
    # 针对CrackForest, PCA_kernel_size(=3)
    theta = cal_cracking_skeleton_orientation_normal(sp, True, -2)
    distances =[]
    for i in range(len(sp)):
        be = draw_cracking_skeleton_orientation_normal(i, sp, theta[i], pic) 
        direction_vector = np.array([be[1][0] - be[0][0], be[1][1] - be[0][1]])
        unit_vector = direction_vector / np.linalg.norm(direction_vector)
        if i % 10==0:
            if (sp[i][1],sp[i][0]) in contours:
                ps = check12p((sp[i][1],sp[i][0]),unit_vector)
                if len(ps)==0:
                    distance = 0.5
                    cv2.circle(pic, (sp[i][1],sp[i][0]), 1,  (0,255,0),1)
                else:
                    if len(ps)==2:
                        a,b = ps[0],ps[1]
                    else:
                        a,b = ps[0],(sp[i][1],sp[i][0])
                    distance = np.linalg.norm(np.array((a[0],a[1]))-np.array((b[0],b[1])))
                    cv2.line(pic, (a[0],a[1]),(b[0],b[1]),(0, 255, 0), 1)
            else:
                #! n=2，HP(ESD)；n=1，OP；n=0, SSD；
                a,b=group_points_by_op(contours,unit_vector, (sp[i][1],sp[i][0]),2)#! 1->2
                if OB:
                    p1, p2 = group_points_by_op(contours,unit_vector, (sp[i][1],sp[i][0]),-1)
                    theta_p1=cal_cracking_skeleton_orientation_normal(p1, True, -2)
                    theta_p2=cal_cracking_skeleton_orientation_normal(p2, True, -2)
                    p11 = [tuple(item) for item in p1]
                    p22 = [tuple(item) for item in p2]
                    if (a[0],a[1]) in p11:
                        for j in range(len(p1)):
                            if p1[j][0]==a[0] and p1[j][1]==a[1]:
                                theta_a = theta_p1[j]
                        for z in range(len(p2)):
                            if p2[z][0]==b[0] and p2[z][1]==b[1]:
                                theta_b = theta_p2[z]
                    else:
                        for j in range(len(p1)):
                            if p1[j][0]==b[0] and p1[j][1]==b[1]:
                                theta_b = theta_p1[j]
                        for z in range(len(p2)):
                            if p2[z][0]==a[0] and p2[z][1]==a[1]:
                                theta_a = theta_p2[z]
                    be = draw_cracking_skeleton_orientation_normal(i, sp, (theta[i]+theta_a+theta_b)/3, pic) 
                    direction_vector = np.array([be[1][0] - be[0][0], be[1][1] - be[0][1]])
                    unit_vector = direction_vector / np.linalg.norm(direction_vector)
                    a,b=group_points_by_op(contours,unit_vector, (sp[i][1],sp[i][0]),2) #! 1->2
                cv2.line(pic, (a[0],a[1]),(b[0],b[1]),(0, 255, 0), 1)
                distance = np.linalg.norm(np.array((a[0],a[1]))-np.array((b[0],b[1])))
            distances.append(distance)
    print("r", np.mean(np.array(distances)))
    sorted_array = np.sort(np.array(distances))
    top_five = sorted_array[-3:]
    average_top_five = np.mean(top_five)
    print("rmax",average_top_five)
    cv2.imshow("Pic", pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()