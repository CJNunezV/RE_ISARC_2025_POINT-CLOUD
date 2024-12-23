import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.core as o3c
import copy
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from pyntcloud import PyntCloud
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# 0) Defining functions

def get_center_line(point_cloud_points):
    centroid = np.mean(point_cloud_points, axis=0)
    pca = PCA(n_components=3)
    pca.fit(point_cloud_points)

    # Dirección principal (primer componente)
    direction = pca.components_[0]  # Vector unitario de la recta

    print("Dirección de la recta:", direction)
    # Punto base: el centroide
    base_point = centroid-(0,0,1.15)

    t = np.linspace(-130, 130, 5201)  # 5cm -> s = 5201 /// 10cm -> 2601
    line_points = base_point + t[:, None] * direction
    line_points_np = np.array(line_points)

    colors_line = np.tile([0,0,0],(line_points_np.shape[0],1))

    line_pc = o3d.t.geometry.PointCloud(line_points_np) 
    line_pc.point.colors = o3c.Tensor(colors_line, dtype=o3c.Dtype.Float32)
    return line_pc,direction,base_point

def delimited_pc(line, point_cloud):
    p1 = line[0]
    p2 = line[-1]
    # Calcula el vector director de la recta
    d = p2 - p1
    d_norm_sq = np.dot(d, d)
    filtered_points = []
    for Q in np.asarray(point_cloud.points):
        t = np.dot(d, Q - p1)
        if 0 <= t <= d_norm_sq:
            filtered_points.append(Q)
    filtered_out= o3d.geometry.PointCloud()
    filtered_out.points = o3d.utility.Vector3dVector(np.array(filtered_points))
    return filtered_out

def distances_pc_line(point_cloud_points,direction,base_point,range):
    direction = direction / np.linalg.norm(direction)
    # Calculamos el vector de cada punto respecto al punto base
    vectors_to_line = point_cloud_points - base_point
    # Producto cruzado entre cada vector y la dirección de la línea
    cross_products = np.cross(vectors_to_line, direction)
    # Magnitud del producto cruzado (distancias al eje)
    distances = np.linalg.norm(cross_products, axis=1)

    mask1 = distances < range  # Índice lógico
    point_cloud_points_in = point_cloud_points[mask1]
    filtered_points_pc_in = o3d.geometry.PointCloud()
    filtered_points_pc_in.points = o3d.utility.Vector3dVector(point_cloud_points_in)

    mask2 = distances >= range  # Índice lógico
    point_cloud_points_out = point_cloud_points[mask2]
    filtered_points_pc_out = o3d.geometry.PointCloud()
    filtered_points_pc_out.points = o3d.utility.Vector3dVector(point_cloud_points_out)
    return distances,filtered_points_pc_in ,filtered_points_pc_out

def deleted_points_out(point_cloud_points, neighbors, max_distance):
    pcd = PyntCloud.from_instance("open3d", point_cloud_points)
    kdtree_id = pcd.add_structure("kdtree")
    k_neighbors = pcd.get_neighbors(k=neighbors, kdtree=kdtree_id) ##List of index of nearest points
    list_index=[]
    for i in range(len(point_cloud_points.points)):
        dis_points =0
        for j in range(len(k_neighbors[i])):
            dis_points += np.linalg.norm(point_cloud_points.points[i] - point_cloud_points.points[k_neighbors[i][j]])
        if(dis_points/neighbors>max_distance):
            list_index.append(i)
    filtered_points = np.asarray(point_cloud_points.points)
    filtered_points = np.delete(filtered_points, list_index, axis=0)

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_pcd
 
def project_point_to_plane(point,point_in_line,vector_line):
    P_minus_P0 = point - point_in_line
    scalar_projection = np.dot(P_minus_P0,vector_line) / np.dot(vector_line,vector_line)
    projection = P - scalar_projection * v 
    return projection

# 1) Variables
print("1) Defining Variables\n")
points_2022 = []
center_geometry_2022 = []

# 2) Importing data
print("2) Importing data")

print("Reading 2022 PC\n")
with open("C:/Users/LENOVO - LAP/Desktop/CHRISTOPHER/RE_ISARC_2025_POINT-CLOUD/Database_Pointcloud/Tunel_2_2019.txt","r") as file:
    for line in file:
      points_2022.append([float(value) for value in line.split()[:3]])
points_2022 = np.array(points_2022)
points_2022[:,0] -= 648000
points_2022[:,1] -= 8566800
points_2022[:,2] -= 2120

my_vowel_size=0.1 #metros
print("El vowel size : 0.10 metros")

# 3) Creating Point Cloud
print("3) Creating Point Cloud - Vowel Size\n")

## Previous configurations
vol = o3d.visualization.SelectionPolygonVolume()
vol= o3d.visualization.read_selection_polygon_volume("D:/ISARC 2025/volumen.json")

print("3.1) Creating a PC based on tensor") ###############################################################################################################################################
pcd_2022_t = o3d.t.geometry.PointCloud()
pcd_2022_t.point["positions"] = o3d.core.Tensor(points_2022, dtype=o3d.core.Dtype.Float32)
print(type(pcd_2022_t))
print(pcd_2022_t)
#o3d.visualization.draw_geometries([pcd_2022_t.to_legacy()], window_name="Nube de puntos 2022 - Tensor") 
print("---Voxel down")
downpcd_2022_t = pcd_2022_t.voxel_down_sample(voxel_size=my_vowel_size)
#o3d.visualization.draw_geometries([downpcd_2022_t.to_legacy()], window_name="Nube de puntos 2022 - Down - Tensor") 
print(type(downpcd_2022_t)) # mantein a tensor feature
print(downpcd_2022_t)
print("---Vertex normal estimation")
downpcd_2022_t.estimate_normals(max_nn=30,radius=0.1)
print(downpcd_2022_t)
normals_t = downpcd_2022_t.point.normals
print(normals_t[0])
print("---Crop PC base on volume")
crop_downpcd_2022_t = vol.crop_point_cloud(downpcd_2022_t.to_legacy())
print(type(crop_downpcd_2022_t))
print(crop_downpcd_2022_t)
print("---Painting point cloud")
downpcd_2022_t.paint_uniform_color([1,0,0])


print("\n\n3.2) Creating a PC based on Geometry 3D") ##########################################################################################################################################
pcd_2022_n = o3d.geometry.PointCloud()
pcd_2022_n.points = o3d.utility.Vector3dVector(points_2022)
print(type(pcd_2022_n))
print(pcd_2022_n)
#o3d.visualization.draw_geometries([pcd_2022_n], window_name="Nube de puntos 2022 - Normal") ################################ PC First
print("---Voxel down")
downpcd_2022_n = pcd_2022_n.voxel_down_sample(voxel_size=my_vowel_size)
deleted_downpcd_2022_n = copy.deepcopy(downpcd_2022_n)
#o3d.visualization.draw_geometries([downpcd_2022_n], window_name="Nube de puntos 2022 - Down - Normal")  ################################ PC Down
print(type(downpcd_2022_n))
print(downpcd_2022_n)
print("---Vertex normal estimation")
downpcd_2022_n.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
print(type(downpcd_2022_n))
print(downpcd_2022_n)
print(downpcd_2022_n.normals[0])
print("---Crop PC base on volume")
crop_downpcd_2022_n = vol.crop_point_cloud(downpcd_2022_n)
print(type(crop_downpcd_2022_n))
print(crop_downpcd_2022_n)
print("---Painting point cloud")
#downpcd_2022_n.paint_uniform_color([1,0,0])


print("\n\n3.3) Eliminación de puntos") #####################################################################################################################################################
print("PRIMERA ITERACIÓN")
deleted_downpcd_2022_n_points = np.asarray(deleted_downpcd_2022_n.points) ##deleted_downpcd_2022_n = copy.deepcopy(downpcd_2022_n)
line_2022_pc,direction,base_point = get_center_line(deleted_downpcd_2022_n_points) ##Input : <class 'numpy.ndarray'>
##Output <class 'open3d.cpu.pybind.t.geometry.PointCloud'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>,<class 'numpy.ndarray'>
distances,filtered_points_pc_in,filtered_points_pc_out = distances_pc_line(deleted_downpcd_2022_n_points,direction,base_point,0.9)

#o3d.visualization.draw_geometries([filtered_points_pc_in,line_2022_pc.to_legacy()])
#o3d.visualization.draw_geometries([filtered_points_pc_out,line_2022_pc.to_legacy()])

print("DELIMITACION DE LONGITUD")
filtered_out= delimited_pc(np.asarray(line_2022_pc.to_legacy().points),filtered_points_pc_out) ##Input <class 'open3d.cpu.pybind.geometry.PointCloud'>

print("SEGUNDA ITERACIÓN")
line_2022_pc_2 ,direction2,base_point2 = get_center_line(np.asarray(filtered_out.points)) ##Input : <class 'numpy.ndarray'>
distances_2,filtered_points_pc_in_2,filtered_points_pc_out_2 = distances_pc_line(np.asarray(filtered_out.points),direction2,base_point2,1)

#o3d.visualization.draw_geometries([filtered_points_pc_in_2,line_2022_pc_2.to_legacy()])
#o3d.visualization.draw_geometries([filtered_points_pc_out_2,line_2022_pc_2.to_legacy()])

print("\n\n3.4) KDTreee")

print("step 1 - Create a Pyntcloud object")
pcd2 = PyntCloud.from_instance("open3d", filtered_points_pc_out_2)
print("cloud has", len(pcd2.points), "points.")

# Find Neighbors
print("Step 2 - Find Neighbors using KD Tree")

k_n = 3
kdtree_id = pcd2.add_structure("kdtree")
k_neighbors = pcd2.get_neighbors(k=k_n, kdtree=kdtree_id) ##List of index of nearest points

point_cloud_clean = deleted_points_out(filtered_points_pc_out_2, k_n, 0.20)
point_cloud_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))

#o3d.visualization.draw_geometries([point_cloud_clean])


# Calculate Eigenvalues for each point
print("Step 3 - Calculate Eigenvalues") ##Calculo de valores propios

ev = pcd2.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

##e3 (mayor): Representa la dirección de mayor varianza en la vecindad de cada punto.
##e2 (intermedio): Varianza en la segunda dirección principal.
##e1 (menor): Dirección con la menor varianza (usualmente asociada a la "normal" de la superficie).

e1 = pcd2.points['e3('+str(k_n+1)+')'].values
e2 = pcd2.points['e2('+str(k_n+1)+')'].values
e3 = pcd2.points['e1('+str(k_n+1)+')'].values

sum_eg = np.add(np.add(e1,e2),e3) ##Suma de los valores propios
sigma = np.divide(e1,sum_eg)

# Convert Points back to Open3D cloud object
print("step 4 - Convert Points back to Open3D cloud")
converted_pcd2 = pcd2.to_instance("open3d", mesh=False)

# Colour points according to the points' first eigenvalues
print("step 5 - Color points according points' first eigenvalues")

cmap = cm.hot
m = cm.ScalarMappable(cmap=cmap)
n = m.to_rgba(sigma)
eigen_colors = np.delete(n,2,1)

converted_pcd2.colors = o3d.utility.Vector3dVector(eigen_colors)

#o3d.visualization.draw_plotly([converted_pcd2], point_sample_factor = 1)

output = ""
print("\n\n3.5) Reconstruction") #####################################################################################################################################################
print("Alpha shapes")
alpha = 20
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud_clean, alpha)
mesh.compute_vertex_normals(normalized=True)
#o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
output += "Alpha Shapes - Alpha: " + str(alpha) + "\n"
output+= str(mesh) + "\n"

print("Ball pivoting")
radii = [0.11, 0.11, 0.11, 0.11]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    point_cloud_clean, o3d.utility.DoubleVector(radii))
#o3d.visualization.draw_geometries([rec_mesh])
output += "Ball pivotin - radii: " + str(radii) + "\n"
output+= str(rec_mesh) + "\n"

print("Poisson surface reconstruction")
depth=14
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as m:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud_clean, depth=depth)
#o3d.visualization.draw_geometries([mesh])
output += "Poisson surface reconstruction - depth: " + str(depth) + "\n"
output+= str(mesh) + "\n"
print(output)

##Impresion de planos cartesianos

import matplotlib.pyplot as plt

# Datos de los puntos
x = [-10, -5, 0, 5, 10]
y = [100, 25, 0, 25, 100]

# Crear el gráfico
plt.figure(figsize=(6, 6))  # Crear un lienzo cuadrado
plt.axhline(0, color='black', linewidth=0.5)  # Eje X
plt.axvline(0, color='black', linewidth=0.5)  # Eje Y
plt.grid(color='gray', linestyle='--', linewidth=0.5)  # Cuadrícula

# Dibujar los puntos
plt.scatter(x, y, color='blue', label='Puntos')

# Dibujar líneas que conecten los puntos
for i in range(len(x) - 1):
    plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color='red', linestyle='-', linewidth=1)

# Personalizar el gráfico
plt.title('Plano Cartesiano con Líneas entre Puntos')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.legend()
plt.show()




##Conclusions
# Voxel Down: Difference in amount of points. 
# Cropping Point Cloud: Result is a Point Cloud base on Geometry 3D always.             