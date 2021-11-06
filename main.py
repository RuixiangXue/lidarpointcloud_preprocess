import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
pcd=o3d.io.read_point_cloud("Ford_01_vox1mm-0100.ply") #orignal voxel size:1mm
print("orignal point cloud:",pcd)

#     preprocess:2 downsample method
#     method1:voxel downsample the point cloud
#caculation: voxelsize=voxel_size*orignal voxel size
downpcd=pcd.voxel_down_sample(voxel_size=1)
#     method2:uniform downsample the point cloud
#     every_k_points=1 means every 1 point sample once
#downpcd = pcd.uniform_down_sample(every_k_points=1)
print("point cloud_after_downsample:",downpcd)

#    preprocess:2 outlier removal method
#    method1:Statistical outlier removal
cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
#    method2:radius outlier removal
#cl, ind = downpcd.remove_radius_outlier(nb_points=16, radius=0.05)
useful_pointcloud = downpcd.select_by_index(ind)
removal_pointcloud =downpcd .select_by_index(ind, invert=True)
print("pointcloud_after_outlier removal:",useful_pointcloud)
o3d.visualization.draw_geometries([removal_pointcloud],window_name='noise&outlier_points')

#uniform color
useful_pointcloud.paint_uniform_color((0,1,0))
#caculate the convex hull
hull, _ = useful_pointcloud.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))
#segment the plane(ground) using RANSAC
plane_model, inliers = useful_pointcloud.segment_plane(distance_threshold=700,ransac_n=10,num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
inlier_cloud = useful_pointcloud.select_by_index(inliers)
inlier_cloud.paint_uniform_color((0, 0, 1))
outlier_cloud = useful_pointcloud.select_by_index(inliers, invert=True)
print("ground point cloud:",inlier_cloud)
print("(exclude ground) point cloud:",outlier_cloud)
o3d.visualization.draw_geometries([outlier_cloud,inlier_cloud],window_name='ground&other_usingRANSAC')
o3d.visualization.draw_geometries([inlier_cloud],window_name='ground_usingRANSAC')
#point cluster using DBSCAN
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        outlier_cloud.cluster_dbscan(eps=2000, min_points=10, print_progress=True))

max_label = labels.max()
print(f"(exclude ground) point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([outlier_cloud],window_name='cluster_result')
#save (exclude ground)data
data_df1=pd.DataFrame(np.c_[np.asarray(outlier_cloud.points),labels])
data_df1.columns=['coordinate_x','coordinate_y','coordinate_z','labels']
writer1=pd.ExcelWriter('outlier.xlsx')
data_df1.to_excel(writer1,'page_1',float_format='%.6f')
writer1.save()
#save  ground data
data_df2=pd.DataFrame(np.c_[np.asarray(inlier_cloud.points)])
data_df2.columns=['coordinate_x','coordinate_y','coordinate_z']
writer2=pd.ExcelWriter('inlier.xlsx')
data_df2.to_excel(writer2,'page_1',float_format='%.6f')
writer2.save()
#save statistics_result of cluster point and draw plot
sorted_label=np.sort(labels,axis=0,kind='quicksort')
element,num=np.unique(sorted_label,return_counts=True)
data_df3=pd.DataFrame(np.c_[element.T,num.T])
data_df3.columns=['element','num']
writer3=pd.ExcelWriter('statistics_result_clusterpoints.xlsx')
data_df3.to_excel(writer3,'page_1',float_format='%.3f')
writer3.save()
plt.plot(element,num)
plt.savefig('element-num.jpg')
#transform all cluster point cloud(include cluseter failure points) into groups of points
cluster_outlier_cloud=np.c_[np.asarray(outlier_cloud.points),labels]
group_cluster=[0 for x in range(max_label+2)]
temp=[0 for x in range(0,np.argmax(group_cluster))]
statistics_labels=np.c_[element.T,num.T]
for y in range(np.size(statistics_labels,0)):
    for j in range(np.size(cluster_outlier_cloud,0)):
        if statistics_labels[y][0]==cluster_outlier_cloud[j][3]:
            temp.append(cluster_outlier_cloud[j,0:3])
    temp=np.array(temp,dtype=object)
    group_cluster[y]=temp
    temp=[]
#visualize the cluster point cloud (using groups of different color and bounding box to make visualization more clear)
alltogether = o3d.geometry.PointCloud()
boundingboxlist=[0 for x in range(len(group_cluster))]
tppointcloud = o3d.geometry.PointCloud()
for k in range(1,len(group_cluster)):
    tppointcloud.points=o3d.utility.Vector3dVector(group_cluster[k])
    boundingboxlist[k] = tppointcloud.get_axis_aligned_bounding_box()
    boundingboxlist[k].color = (1, 0, 0)
    alltogether = alltogether + tppointcloud
linesetbox=o3d.geometry.LineSet()
for x in range(1,len(group_cluster)):
    linesetbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(boundingboxlist[x])+linesetbox
o3d.visualization.draw_geometries([alltogether,linesetbox],window_name='group_cluster_points')
#visualize the cluster failure point cloud
tppointcloud.points=o3d.utility.Vector3dVector(group_cluster[0])
o3d.visualization.draw_geometries([tppointcloud],window_name='cluster_failure_points')
print("cluster failure point cloud",tppointcloud)
# after above process step,pointcloud(afterdownsample) can be divided into
# 1.group_cluster type:list of array index[0]:cluster failure points group index[other] cluster successful points group
# 2.inlier_cloud  type:pointcloud    groud points
# 3.removal_pointcloud type:pointcloud    filtered point cloud using outlier_removal
# transform cluster point(exclude cluster failure points) into 3D representation








