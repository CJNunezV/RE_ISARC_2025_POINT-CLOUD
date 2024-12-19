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

    t = np.linspace(-130, 130, 2601)  # 5cm -> s = 5201 /// 10cm -> 2601   /// 20cm -> 1301
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
    projection = point - scalar_projection * vector_line 
    return projection

def transform_to_xy(point,line,point_line):
    L_unit = line / np.linalg.norm(line)
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
    transformed_point = np.dot(R, point - point_line)
    return transformed_point[:2]

def rotate(point, angle):
    angle_radians = np.radians(angle)

    # Matriz de rotación
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    rotated_point = np.dot(rotation_matrix, point)
    return rotated_point

def order_points_by_proximity(points):
    # Convertir la lista de puntos en un array de NumPy
    points = np.array(points)
    
    # Inicializar la lista de puntos ordenados
    ordered_points = []
    
    # Seleccionar el primer punto más cercano al origen (0, 0)
    current_point = points[np.argmin(np.linalg.norm(points, axis=1))]
    ordered_points.append(current_point)
    
    # Eliminar el punto seleccionado de la lista original
    remaining_points = points[~np.all(points == current_point, axis=1)]
    
    # Iterar hasta que no queden puntos
    while len(remaining_points) > 0:
        # Calcular la distancia desde el punto actual a todos los restantes
        distances = np.linalg.norm(remaining_points - current_point, axis=1)
        
        # Seleccionar el punto más cercano
        current_point = remaining_points[np.argmin(distances)]
        ordered_points.append(current_point)
        
        # Eliminar el punto seleccionado de los puntos restantes
        remaining_points = remaining_points[~np.all(remaining_points == current_point, axis=1)]
    
    return np.array(ordered_points)

#Comments
#pcd_2022_n = pc base
#downpcd_2022_n = down pc base
#filtered_points_pc_out_2 =  clean(floor - mid) down pc base
#point_cloud_clean = clean (all) down pc base


# 1) Variables
print("1) Defining Variables\n")
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

print("Reading 2022 PC\n")
with open("D:/ISARC 2025/2022\Tunel2/2. Nube de puntos/Tunel_2_2022.txt","r") as file:
    for line in file:
      points_2022.append([float(value) for value in line.split()[:3]])
points_2022 = np.array(points_2022)
points_2022[:,0] -= 648000
points_2022[:,1] -= 8566800
points_2022[:,2] -= 2120

print("Reading 2024 PC")
path_pcd_2024 = "D:/ISARC 2025/2024/Tunel_2_2024.pts"
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
pcd_2022_n = o3d.geometry.PointCloud()
pcd_2022_n.points = o3d.utility.Vector3dVector(points_2022)
#o3d.visualization.draw_geometries([pcd_2022_n], window_name="Nube de puntos 2022 - Normal") ################################ PC First
print("---Voxel down")
downpcd_2022_n = pcd_2022_n.voxel_down_sample(voxel_size=my_vowel_size)
deleted_downpcd_2022_n = copy.deepcopy(downpcd_2022_n)
#o3d.visualization.draw_geometries([downpcd_2022_n], window_name="Nube de puntos 2022 - Down - Normal")  ################################ PC Down
print("---Vertex normal estimation")
downpcd_2022_n.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
print("---Painting point cloud")
#downpcd_2022_n.paint_uniform_color([1,0,0])


print("\n\n3.2) Eliminación de puntos") #####################################################################################################################################################
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

#Resultado de esta interacción : filtered_points_pc_out_2


print("\n\n3.3) KDTreee")
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


print("\n\n3.4) Getting Cross-Sections")
#PC: point_cloud_clean
#Line: line_2022_pc_2 'open3d.cpu.pybind.t.geometry.PointCloud'>
#Direc: direction2
points_clean = point_cloud_clean.points

line_numpy = line_2022_pc_2.point.positions.numpy()

for i in range(len(line_numpy) - 1):
    pair = np.array([line_numpy[i], line_numpy[i + 1]])  
    section_points = delimited_pc(pair, point_cloud_clean) #<class 'numpy.ndarray'>
    list_points=[]
    list_x=[]
    list_y=[]
    for point in np.asarray(section_points.points):
        #projection = project_point_to_plane(point, line_numpy[i], direction2)
        #print(projection)
        projection = rotate(transform_to_xy(point,direction2, line_numpy[i]),-10)
        list_points.append(projection)
    
    list_points_order= order_points_by_proximity(list_points)
    list_points_order = np.vstack([list_points_order, list_points_order[0]])
    
    for i in list_points_order:
        list_x.append(i[0])
        list_y.append(-i[1]) #Debido a la data obtenida, se multiplica por -1
    plt.scatter(list_x, list_y, c='blue', marker='o', label='Puntos') 
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Profile +0.00')
    plt.grid(True)
    for i in range(len(list_x) - 1):
        plt.plot([list_x[i], list_x[i + 1]], [list_y[i], list_y[i + 1]], color='red', linestyle='-', linewidth=2)
    plt.legend()
    plt.show()
    o3d.visualization.draw_geometries([section_points])


print("\n\n3.5) Reconstruction") #####################################################################################################################################################
print("Alpha shapes")
alpha = 20
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud_clean, alpha)
mesh.compute_vertex_normals(normalized=True)
#o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


print ("\n\n3.5) Cross-Sectional Tunnel")
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
#plt.show()




##Conclusions
# Voxel Down: Difference in amount of points. 
# Cropping Point Cloud: Result is a Point Cloud base on Geometry 3D always.             

##Para el caso de tensores
##"name_point_base_tensor".point = obtienes todos los atributos que tienen los puntos
##"name_point_base_tensor".point."name_atributo" = podras ver <class 'open3d.cpu.pybind.core.Tensor'>
##Entonces tienes que convertirlo a numpy ".numpy()"
