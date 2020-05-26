
# 文件功能：
# ransac分割地面点云

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.cluster import KMeans
import open3d as o3d
import random

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data,normals):
    # 作业1
    # 屏蔽开始
    p = 0.99    #识别到模型的期望概率
    points_num = data.shape[0]      #点云点数
    distance_threshold = 0.3         #RANSAC内点距离阈值
    best_inlier_num = -1       #最多内点数
    max_iter = 200      #最大迭代次数
    # k = max_iter
    i = 0
    while i < max_iter:
        i+=1
        #随机选三个点
        # rand_index = np.random.choice(indices,3,replace=False)
        random_sample = random.sample(range(points_num), 3)      #sample比choise速度快

        sample_points = data[random_sample,:]
        # 三点共线检测
        if np.linalg.matrix_rank(sample_points) <3:
            continue
        p0 = data[random_sample[0], :]
        p1 = data[random_sample[1], :]
        p2 = data[random_sample[2], :]
        # print('random points:',p0, p1, p2)
        n0 = normals[random_sample[0], :]
        n1 = normals[random_sample[1], :]
        n2 = normals[random_sample[2], :]

        if ((np.abs(np.dot(n0,n1))>0.95) & (np.abs(np.dot(n0,n2))>0.95)):
            # ss = ss + 1
            # 计算平面方程
            v0 = p0 - p1
            v1 = p1 - p2
            plane_normal = np.cross(v0,v1)
            plane_normal = plane_normal/np.linalg.norm(plane_normal)
            # plane_normal_norm2 = np.linalg.norm(plane_normal)

            dis = np.abs(np.dot((data - p0),plane_normal))

            inlier_fit = np.array(dis < distance_threshold)
            inlier_size = np.sum(inlier_fit, axis=0)
            if inlier_size > best_inlier_num:
                best_inlier_num = inlier_size
                best_plane_normal = plane_normal
                print (plane_normal)
                inlier_indices = np.argwhere(inlier_fit==True).flatten()
                outlier_indices = np.argwhere(inlier_fit==False).flatten()

                # 最大迭代次数K，是P的函数, 因为迭代次数足够多，就一定能找到最佳的模型，
                # P是模型提供理想需要结果的概率，也可以理解为模型的点都是内点的概率？
                # P = 0.99
                # 当前模型，所有点中随机抽取一个点，它是内点的概率
                w = best_inlier_num / points_num
                # np.power(w, 3)是随机抽取三个点，都是内点的概率
                # 1-np.power(w, 3)，就是三个点至少有一个是外点的概率，
                # 也就是得到一个坏模型的概率;
                # 1-P 代表模型永远不会选出一个3个点都是内点的集合的概率，
                # 也就是至少有一个外点；
                # K 就是得到的需要尝试多少次才会得到当前模型的理论次数
                # TODO 完善对该理论最大论迭代次数的理解
                k = (np.log(1-p) / np.log(1.0 - np.power(w, 3))) #+ sd_w
                if k < max_iter:        #如果当前最大迭代次数max_iter小于理论计算值k，则用k替代
                    max_iter = k

    # TODO 这是是否还有一个重新拟合模型的步骤，
    #   有的话怎么用多个点求一个平面

    #
    inlier_cloud = data[inlier_indices]
    outlier_cloud = data[outlier_indices]
    # print (inlier_cloud.shape)
    #
    # # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    # # pcd = o3d.geometry.PointCloud()
    # # pcd.points = o3d.utility.Vector3dVector(inlier_cloud)
    # # o3d.io.write_point_cloud("ground.pcd", pcd)
    # # pcd.points = o3d.utility.Vector3dVector(outlier_cloud)
    # # o3d.io.write_point_cloud("segmengted.pcd", pcd)
    #
    # # 屏蔽结束
    #
    # print('origin data points num:', data.shape[0])
    # print('segmented data points num:', outlier_cloud.shape[0])
    return inlier_cloud, outlier_cloud


def colored(data,color):
    colors = np.zeros(data.shape)
    for i in range(len(data)):
        colors[i,:] = color
    return colors


if __name__ == '__main__':
    pcd = o3d.geometry.PointCloud()
    pcd = o3d.io.read_point_cloud("000001.pcd")
    # 计算法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=1, max_nn=30))

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    time_start = time.time()

    ground, seg = ground_segmentation(points,normals)
    time_end = time.time()
    print('totally cost', time_end - time_start)

    # # colored
    ground_o3d = o3d.geometry.PointCloud()
    ground_o3d.points = o3d.utility.Vector3dVector(ground)
    ground_o3d.colors = o3d.utility.Vector3dVector(colored(ground, [0, 0, 1]))
    #
    seg_o3d = o3d.geometry.PointCloud()
    seg_o3d.points = o3d.utility.Vector3dVector(seg)
    seg_o3d.colors = o3d.utility.Vector3dVector(colored(seg, [1, 0, 0]))
    #
    o3d.visualization.draw_geometries([ground_o3d,seg_o3d])

    # save cloud
    # o3d.io.write_point_cloud("inlier.pcd", ground_o3d)
    # o3d.io.write_point_cloud("outlier.pcd", seg_o3d)

