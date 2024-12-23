import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.core as o3c
import os
import copy
import sys

# 1) Variables
print("1) Defining Variables")
points_2019 = []
points_2022 = []
points_2024 = []

# 2) Importing data
print("2) Importing data")

print("Reading 2019 PC")
with open("D:/ISARC 2025/2019/Tunel_2_2019.txt","r") as file:
    for line in file:
      points_2019.append([float(value) for value in line.split()[:3]])
points_2019 = np.array(points_2019)
points_2019[:,0] -= 648000
points_2019[:,1] -= 8566800
points_2019[:,2] -= 2120
print("El primer punto del 2019 es : ",points_2019[1])

print("Reading 2022 PC")
with open("D:/ISARC 2025/2022\Tunel2/2. Nube de puntos/Tunel_2_2022.txt","r") as file:
    for line in file:
      points_2022.append([float(value) for value in line.split()[:3]])
points_2022 = np.array(points_2022)
points_2022[:,0] -= 648000
points_2022[:,1] -= 8566800
points_2022[:,2] -= 2120
print("El primer punto del 2022 es : ", points_2022[1])

print("Reading 2024 PC")
path_pcd_2024 = "D:/ISARC 2025/2024/Tunel_2_2024.pts"
pcd_2024_v = o3d.io.read_point_cloud(path_pcd_2024)

#with open("C:/Users/CNUNEZ/Downloads/DATA-CLIENTE/20200214/Tunel01/Muestra - 25%.txt","r") as file:
#    for line in file:
#      points_2024.append([float(value)/1000 for value in line.split()])

points_2024 = np.asarray(pcd_2024_v.points)
points_2024[:,0] -= 648000
points_2024[:,1] -= 8566800
points_2024[:,2] -= 2120

print("El primer punto del 2024 es ", points_2024[0])

my_vowel_size=0.1 #metros
print("El vowel size : 0.10 metros")


print("3) Creating Point Cloud - Vowel Size")

##<class 'open3d.cpu.pybind.geometry.PointCloud'>
##<class 'open3d.cpu.pybind.t.geometry.PointCloud'>
pcd_2019 = o3d.t.geometry.PointCloud()
pcd_2019 = o3d.t.geometry.PointCloud(points_2019)
downpcd_2019=pcd_2019.voxel_down_sample(voxel_size=my_vowel_size)
print("Downpcd 2019 = ",downpcd_2019)
positions_2019 = downpcd_2019.point.positions.numpy()

pcd_2022 = o3d.t.geometry.PointCloud()
pcd_2022 = o3d.t.geometry.PointCloud(points_2022)
downpcd_2022=pcd_2022.voxel_down_sample(voxel_size=my_vowel_size)
print("Downpcd 2022 = ",downpcd_2022)
positions_2022 = downpcd_2022.point.positions.numpy()

pcd_2024 = o3d.t.geometry.PointCloud()
pcd_2024 = o3d.t.geometry.PointCloud(points_2024)
downpcd_2024=pcd_2024.voxel_down_sample(voxel_size=my_vowel_size)
print("Downpcd 2024 = ",downpcd_2024)
positions_2024 = downpcd_2024.point.positions.numpy()

pcd_final = o3d.t.geometry.PointCloud()

number_2022 = positions_2022.shape[0]
number_2019 = positions_2019.shape[0]
number_2024 = positions_2024.shape[0]
colors_2019 = np.tile([1,0,0],(number_2019,1))
colors_2022 = np.tile([0,0,1],(number_2022,1))
colors_2024 = np.tile([0,1,0],(number_2024,1))


print("4) Combining point clouds")
points_final = np.concatenate((positions_2019, positions_2022,positions_2024))
print(points_final.shape)
colors_final = np.concatenate((colors_2019, colors_2022,colors_2024))

points_final = o3d.t.geometry.PointCloud(points_final)
points_final.point.colors = o3c.Tensor(colors_final, dtype=o3c.Dtype.Float32)

#o3d.visualization.draw_plotly([points_final.to_legacy()],point_sample_factor=0.3,window_name="Tunnels")
o3d.visualization.draw_geometries([points_final.to_legacy()],window_name="Tunnels") 

##Normal
print("Normal")

copy_downpcd_2024 = copy.deepcopy(downpcd_2024.to_legacy())
## para tensor : copy_downpcd_2024.estimate_normals(max_nn=30, radius=0.1)
print(type(copy_downpcd_2024))
copy_downpcd_2024.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#o3d.visualization.draw_geometries([copy_downpcd_2024.to_legacy()], point_show_normal=True, window_name="Normal")
print(type(pcd_2019))
print(type(pcd_2022))
print(type(pcd_2024))
print(type(points_final))
print(type(copy_downpcd_2024))


#tiene_normales = copy_downpcd_2024.has_normals()
#print(f"La nube de puntos tiene normales: {tiene_normales}")

##Creating a mesh from  Poisson
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
  mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(copy_downpcd_2024, depth=9) ## Solo funciona con el point cloud normal (no tensor)
bbox = copy_downpcd_2024.get_axis_aligned_bounding_box()
mesh_poisson_crop = mesh_poisson.crop(bbox)
#o3d.visualization.draw_geometries([mesh_poisson_crop], mesh_show_back_face=True)

##Creating Ball pivotin

radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(copy_downpcd_2024, o3d.utility.DoubleVector(radii)) ## Solo funciona con el point cloud normal (no tensor)
#o3d.visualization.draw_geometries([copy_downpcd_2024, rec_mesh])