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
def get_center_line(point_cloud_points_f,offset):
    centroid = np.mean(point_cloud_points_f, axis=0)
    pca = PCA(n_components=3)
    pca.fit(point_cloud_points_f)

    # Dirección principal (primer componente)
    direction_tunnel_1_f = pca.components_[0]  # Vector unitario de la recta

    print("Dirección de la recta:", direction_tunnel_1_f)
    # Punto base: el centroide
    base_point_tunnel_1_f = centroid-(0,0,offset)

    t = np.linspace(-130, 130, 2601)  # 5cm -> s = 5201 /// 10cm -> 2601   /// 20cm -> 1301
    line_points_f = base_point_tunnel_1_f + t[:, None] * direction_tunnel_1_f
    line_points_np_f = np.array(line_points_f)

    colors_line = np.tile([0,0,0],(line_points_np_f.shape[0],1))

    line_pc_f = o3d.t.geometry.PointCloud(line_points_np_f) 
    line_pc_f.point.colors = o3c.Tensor(colors_line, dtype=o3c.Dtype.Float32)
    return line_pc_f,direction_tunnel_1_f,base_point_tunnel_1_f
##Input : <class 'numpy.ndarray'>
##Output <class 'open3d.cpu.pybind.t.geometry.PointCloud'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>,<class 'numpy.ndarray'>

def delimited_pc(line_f, point_cloud_f):
    p1 = line_f[0]
    p2 = line_f[-1]
    # Calcula el vector director de la recta
    d = p2 - p1
    d_norm_sq = np.dot(d, d)
    filtered_points_f = []
    for Q in np.asarray(point_cloud_f.points):
        t = np.dot(d, Q - p1)
        if 0 <= t <= d_norm_sq:
            filtered_points_f.append(Q)
    filtered_f= o3d.geometry.PointCloud()
    filtered_f.points = o3d.utility.Vector3dVector(np.array(filtered_points_f))
    return filtered_f
##Input <class 'open3d.cpu.pybind.geometry.PointCloud'>

def distances_pc_line(point_cloud_points_f,direction_tunnel_1_f,base_point_tunnel_1_f,range):
    direction_tunnel_1_f = direction_tunnel_1_f / np.linalg.norm(direction_tunnel_1_f)
    vectors_to_line = point_cloud_points_f - base_point_tunnel_1_f
    cross_products = np.cross(vectors_to_line, direction_tunnel_1_f)
    distances_f = np.linalg.norm(cross_products, axis=1)

    mask1 = distances_f < range
    point_cloud_points_in_f = point_cloud_points_f[mask1]
    filtered_points_pc_in_f = o3d.geometry.PointCloud()
    filtered_points_pc_in_f.points = o3d.utility.Vector3dVector(point_cloud_points_in_f)

    mask2 = distances_f >= range
    point_cloud_points_out_f = point_cloud_points_f[mask2]
    filtered_points_pc_out_2022_f = o3d.geometry.PointCloud()
    filtered_points_pc_out_2022_f.points = o3d.utility.Vector3dVector(point_cloud_points_out_f)
    return distances_f,filtered_points_pc_in_f ,filtered_points_pc_out_2022_f

def deleted_points_out(point_cloud_points, neighbors, max_distance):
    pcd = PyntCloud.from_instance("open3d", point_cloud_points)
    kdtree_2022_id = pcd.add_structure("kdtree")
    k_neighbors = pcd.get_neighbors(k=neighbors, kdtree=kdtree_2022_id) ##List of index of nearest points
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
 
def project_point_to_plane(point_f,point_in_line_f,vector_line):
    P_minus_P0 = point_f - point_in_line_f
    scalar_projection = np.dot(P_minus_P0,vector_line) / np.dot(vector_line,vector_line)
    projection = point_f - scalar_projection * vector_line 
    return projection

def transform_to_xy(point_f,line_f,point_line_f):
    L_unit = line_f / np.linalg.norm(line_f)
    Z_axis = np.array([0, 0, 1])
    k = np.cross(L_unit, Z_axis)
    if np.linalg.norm(k) == 0:
        R = np.eye(3) 
    else:
        k_unit = k / np.linalg.norm(k)
        theta = np.arccos(np.dot(L_unit, Z_axis))
        K = np.array([
            [0, -k_unit[2], k_unit[1]],
            [k_unit[2], 0, -k_unit[0]],
            [-k_unit[1], k_unit[0], 0]
        ])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    transformed_point = np.dot(R, point_f - point_line_f)
    return transformed_point[:2]

def rotate(point_f, angle_f):
    angle_radians = np.radians(angle_f)

    # Matriz de rotación
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    rotated_point_f = np.dot(rotation_matrix, point_f)
    return rotated_point_f

def order_points_by_proximity(points_f):
    points_f = np.array(points_f)
    
    ordered_points_f = []
    
    current_point = points_f[np.argmin(np.linalg.norm(points_f, axis=1))]
    ordered_points_f.append(current_point)
    
    remaining_points = points_f[~np.all(points_f == current_point, axis=1)]
    
    while len(remaining_points) > 0:
        distances = np.linalg.norm(remaining_points - current_point, axis=1)
        current_point = remaining_points[np.argmin(distances)]
        ordered_points_f.append(current_point)
        remaining_points = remaining_points[~np.all(remaining_points == current_point, axis=1)]
    
    return np.array(ordered_points_f)

#Comments
#pcd_2022_n = pc base
#downpcd_2022_n = down pc base
#filtered_2022_n =  clean(floor - mid) down pc base
#clean_2022 = clean (all) down pc base


# 1) Variables
print("1) Defining Variables\n")
points_2019 = []
points_2022 = []
points_2024 = []


# 2) Importing data
print("2) Importing data")

print("Reading 2019 PC")
with open("C:/Users/LENOVO - LAP/Desktop/CHRISTOPHER/RE_ISARC_2025_POINT-CLOUD/Database_Pointcloud/Tunel_2_2019.txt","r") as file:
    for line in file:
      points_2019.append([float(value) for value in line.split()[:3]])
points_2019 = np.array(points_2019)
points_2019[:,0] -= 648000
points_2019[:,1] -= 8566800
points_2019[:,2] -= 2120

print("Reading 2022 PC")
with open("C:/Users/LENOVO - LAP/Desktop/CHRISTOPHER/RE_ISARC_2025_POINT-CLOUD/Database_Pointcloud/Tunel_2_2022.txt","r") as file:
    for line in file:
      points_2022.append([float(value) for value in line.split()[:3]])
points_2022 = np.array(points_2022)
points_2022[:,0] -= 648000
points_2022[:,1] -= 8566800
points_2022[:,2] -= 2120

print("Reading 2024 PC")
path_pcd_2024 = "C:/Users/LENOVO - LAP/Desktop/CHRISTOPHER/RE_ISARC_2025_POINT-CLOUD/Database_Pointcloud/Tunel_2_2024.pts"
pcd_2024_v = o3d.io.read_point_cloud(path_pcd_2024)
points_2024 = np.asarray(pcd_2024_v.points)
points_2024[:,0] -= 648000
points_2024[:,1] -= 8566800
points_2024[:,2] -= 2120

my_vowel_size=0.1 #metros
print("El vowel size : 0.10 metros")

# 3) Creating Point Cloud
print("3) Creating Point Clouds\n")

print("\n\n3.1) Creating a PC based on Geometry 3D") ##########################################################################################################################################
pcd_2019_n = o3d.geometry.PointCloud()
pcd_2022_n = o3d.geometry.PointCloud()
pcd_2024_n = o3d.geometry.PointCloud()

pcd_2019_n.points = o3d.utility.Vector3dVector(points_2019)
pcd_2022_n.points = o3d.utility.Vector3dVector(points_2022)
pcd_2024_n.points = o3d.utility.Vector3dVector(points_2024)

#o3d.visualization.draw_geometries([pcd_2022_n], window_name="Nube de puntos 2022 - Normal") ################################ PC First
print("---Voxel down")
downpcd_2019_n = pcd_2019_n.voxel_down_sample(voxel_size=my_vowel_size)
downpcd_2022_n = pcd_2022_n.voxel_down_sample(voxel_size=my_vowel_size)
downpcd_2024_n = pcd_2024_n.voxel_down_sample(voxel_size=my_vowel_size)

#o3d.visualization.draw_geometries([downpcd_2024_n], window_name="Nube de puntos 2022 - Down - Normal")  ################################ PC Down

print("---Getting a copy")
deleted_downpcd_2019_n = copy.deepcopy(downpcd_2019_n)
deleted_downpcd_2022_n = copy.deepcopy(downpcd_2022_n)
deleted_downpcd_2024_n = copy.deepcopy(downpcd_2024_n)

print("---Vertex normal estimation")
downpcd_2019_n.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
downpcd_2022_n.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
downpcd_2024_n.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))


print("\n\n3.2) Eliminación de puntos") #####################################################################################################################################################
print("PRIMERA ITERACIÓN")
deleted_downpcd_2019_n_points = np.asarray(deleted_downpcd_2019_n.points)
deleted_downpcd_2022_n_points = np.asarray(deleted_downpcd_2022_n.points)
deleted_downpcd_2024_n_points = np.asarray(deleted_downpcd_2024_n.points) 

center_line_tunnel_1,direction_tunnel_1,base_point_tunnel_1 = get_center_line(deleted_downpcd_2022_n_points,1.15) 

distances_2019,filtered_points_pc_in_2019,filtered_points_pc_out_2019 = distances_pc_line(deleted_downpcd_2019_n_points,direction_tunnel_1,base_point_tunnel_1,0.9)
distances_2022,filtered_points_pc_in_2022,filtered_points_pc_out_2022 = distances_pc_line(deleted_downpcd_2022_n_points,direction_tunnel_1,base_point_tunnel_1,0.9)
distances_2024,filtered_points_pc_in_2024,filtered_points_pc_out_2024 = distances_pc_line(deleted_downpcd_2024_n_points,direction_tunnel_1,base_point_tunnel_1,0.7)

#o3d.visualization.draw_geometries([filtered_points_pc_in_2019,center_line_tunnel_1.to_legacy()])
#o3d.visualization.draw_geometries([filtered_points_pc_in_2024,center_line_tunnel_1.to_legacy()])

print("DELIMITACION DE LONGITUD")
filtered_2019_out= delimited_pc(np.asarray(center_line_tunnel_1.to_legacy().points),filtered_points_pc_out_2019)
filtered_2022_out= delimited_pc(np.asarray(center_line_tunnel_1.to_legacy().points),filtered_points_pc_out_2022)
filtered_2024_out= delimited_pc(np.asarray(center_line_tunnel_1.to_legacy().points),filtered_points_pc_out_2024)


print("SEGUNDA ITERACIÓN")
center_line_tunnel_2 ,direction_tunnel_2,base_point_tunnel_2 = get_center_line(np.asarray(filtered_2022_out.points),1.10)

print("POINT CLOUD LIMPIOS")
distances_2019,filtered_points_pc_in_2019,filtered_2019_n = distances_pc_line(np.asarray(filtered_2019_out.points),direction_tunnel_2,base_point_tunnel_2,1.05)
distances_2022,filtered_points_pc_in_2022,filtered_2022_n = distances_pc_line(np.asarray(filtered_2022_out.points),direction_tunnel_2,base_point_tunnel_2,1)
distances_2024,filtered_points_pc_in_2024,filtered_2024_n = distances_pc_line(np.asarray(filtered_2024_out.points),direction_tunnel_2,base_point_tunnel_2,0.9)

#o3d.visualization.draw_geometries([filtered_points_pc_in_2022,center_line_tunnel_2.to_legacy()])
#o3d.visualization.draw_geometries([filtered_points_pc_in_2024,center_line_tunnel_2.to_legacy()])


print("\n\n3.3) KDTreee")
print("step 1 - Create a Pyntcloud object")
pynt_2019 = PyntCloud.from_instance("open3d", filtered_2019_n)
print("Point cloud 2019 has", len(pynt_2019.points), "points.")

pynt_2022 = PyntCloud.from_instance("open3d", filtered_2022_n)
print("Point cloud 2022 has", len(pynt_2022.points), "points.")

pynt_2024 = PyntCloud.from_instance("open3d", filtered_2024_n)
print("Point cloud 2024 has", len(pynt_2024.points), "points.")

# Find Neighbors
print("Step 2 - Find Neighbors using KD Tree")
k_n = 3

clean_2019 = deleted_points_out(filtered_2019_n, k_n, 0.20)
clean_2022 = deleted_points_out(filtered_2022_n, k_n, 0.20)
clean_2024 = deleted_points_out(filtered_2024_n, k_n, 0.20)

clean_2019.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
clean_2022.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
clean_2024.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
#o3d.visualization.draw_geometries([clean_2019])


print("\n\n3.4) Getting Cross-Sections")
#PC: clean_2022
#Line: center_line_tunnel_2 'open3d.cpu.pybind.t.geometry.PointCloud'>
#Direc: direction_tunnel_2

center_line_numpy = center_line_tunnel_2.point.positions.numpy()

for i in range(len(center_line_numpy) - 1):
    pair = np.array([center_line_numpy[i], center_line_numpy[i + 1]])  

    section_points_2019 = delimited_pc(pair, clean_2019)
    section_points_2022 = delimited_pc(pair, clean_2022)
    section_points_2024 = delimited_pc(pair, clean_2024)

    list_points_2019=[]
    list_points_2022=[]
    list_points_2024=[]

    list_x_2019=[]
    list_x_2022=[]
    list_x_2024=[]

    list_y_2019=[]
    list_y_2022=[]
    list_y_2024=[]

    for point_2019 in np.asarray(section_points_2019.points):
        projection_2019 = rotate(transform_to_xy(point_2019,direction_tunnel_2, center_line_numpy[i]),-10)
        list_points_2019.append(projection_2019)
    for point_2022 in np.asarray(section_points_2022.points):
        projection_2022 = rotate(transform_to_xy(point_2022,direction_tunnel_2, center_line_numpy[i]),-10)
        list_points_2022.append(projection_2022)
    for point_2024 in np.asarray(section_points_2024.points):
        projection_2024 = rotate(transform_to_xy(point_2024,direction_tunnel_2, center_line_numpy[i]),-10)
        list_points_2024.append(projection_2024)
    
    list_points_order_2019 = order_points_by_proximity(list_points_2019)
    list_points_order_2022 = order_points_by_proximity(list_points_2022)
    list_points_order_2024 = order_points_by_proximity(list_points_2024)

    #list_points_order_2019 = np.vstack([list_points_order_2019, list_points_order_2019[0]])
    #list_points_order_2022 = np.vstack([list_points_order_2022, list_points_order_2022[0]])
    #list_points_order_2024 = np.vstack([list_points_order_2024, list_points_order_2024[0]])

    for j in list_points_order_2019:
        list_x_2019.append(j[0])
        list_y_2019.append(-j[1])
    for i in list_points_order_2022:
        list_x_2022.append(i[0])
        list_y_2022.append(-i[1]) #Debido a la data obtenida, se multiplica por -1
    for k in list_points_order_2024:
        list_x_2024.append(k[0])
        list_y_2024.append(-k[1])    
    
    plt.plot(list_x_2019, list_y_2019, color='blue', linestyle='-', label='2019')
    plt.plot(list_x_2022, list_y_2022, color='green', linestyle='-', label='2022')
    plt.plot(list_x_2024, list_y_2024, color='red', linestyle='-', label='2024')
    plt.title('ISARC - 2025')

    # Añadir una leyenda
    plt.legend(title='Años', loc='upper left')

    # Etiquetas de los ejes
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')

    plt.show()
    #o3d.visualization.draw_geometries([section_points_2022])


print("\n\n3.5) Reconstruction") #####################################################################################################################################################
print("Alpha shapes")
alpha = 20
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(clean_2022, alpha)
mesh.compute_vertex_normals(normalized=True)
#o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)






##Conclusions
# Voxel Down: Difference in amount of points. 
# Cropping Point Cloud: Result is a Point Cloud base on Geometry 3D always.             

##Para el caso de tensores
##"name_point_base_tensor".point = obtienes todos los atributos que tienen los puntos
##"name_point_base_tensor".point."name_atributo" = podras ver <class 'open3d.cpu.pybind.core.Tensor'>
##Entonces tienes que convertirlo a numpy ".numpy()"
