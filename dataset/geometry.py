import numpy as np
from copy import deepcopy

import pdb

class Geometry(object):
    def __init__(self, config):
        self.v_res = config['nVoxel'][0]    # ct scan
        self.p_res = config['nDetector'][0] # projections
        self.v_spacing = np.array(config['dVoxel'])[0]    # mm
        self.p_spacing = np.array(config['dDetector'])[0] # mm
        # NOTE: only (res * spacing) is used

        self.DSO = config['DSO'] # mm, source to origin
        self.DSD = config['DSD'] # mm, source to detector
    def coeff(self,points, angle):
        d1 = self.DSO
        d2 = self.DSD

        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        points *= self.v_res * self.v_spacing # mm

        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [            0,              0, 1]
        ])
        points = points @ rot_M.T
        
        coeff = (d2) / (d1 - points[:, 0]) # N,

        return coeff


    def project(self, points, angle):
        # points: [N, 3] ranging from [0, 1]
        # d_points: [N, 2] ranging from [-1, 1]
        #pdb.set_trace()
        d1 = self.DSO
        d2 = self.DSD

        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        points *= self.v_res * self.v_spacing # mm

        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [            0,              0, 1]
        ])
        points = points @ rot_M.T
        
        coeff = (d2) / (d1 - points[:, 0]) # N,
        #pdb.set_trace()
        d_points = points[:, [2, 1]] * coeff[:, None] # [N, 2] float
        d_points /= (self.p_res * self.p_spacing)
        d_points *= 2 # NOTE: some points may fall outside [-1, 1]
        #pdb.set_trace()

        return d_points
    def caculate_spatial_attribute(self, points, angles):
        #* input 
        #* points shape [n,3]
        #* angles [0,1.5]
        d1 = self.DSO
        d2 = self.DSD

        #* 转换到ct坐标系下 
        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        points *= self.v_res * self.v_spacing # mm

        # 方法1：固定点，计算不同角度下源的位置
        views_spatial_attribute = []
            
    
        for angle in angles:
            # 计算该角度下源在世界坐标系中的位置
            # 源绕原点旋转
            attribute_list = []
            source_x = d1 * np.cos(angle)
            source_y = d1 * np.sin(angle)
            source_position = np.array([source_x, source_y, 0])
            
            # 计算点到源的距离
            #dis_vector = points - source_position
            #distances = np.linalg.norm(dis_vector, axis=1)

            #distances_norm = (distances - distances.min()) / (distances.max() - distances.min())
            #pdb.set_trace()
            #z_component = dis_vector[:,2]
            #xy_component = np.sqrt(dis_vector[:,0]**2 + dis_vector[:,1]**2)
            #elevation_angles_rad = np.arctan2(z_component , xy_component)
            #elevation_angles_rad_norm = ( elevation_angles_rad + np.pi/2) / np.pi
            
            #pdb.set_trace()
            # coeff 
            angle_inv = -1 * angle
            rot_M = np.array([
                [np.cos(angle_inv), -np.sin(angle_inv), 0],
                [np.sin(angle_inv),  np.cos(angle_inv), 0],
                [            0,              0, 1]
            ])
            rotated_points = points @ rot_M.T
            coeff = d2 / (d1 - rotated_points[:, 0])
            #pdb.set_trace()
            coeff_log = np.log(coeff)  # 将乘法关系转换为加法关系
            coeff_log_norm = (coeff_log - coeff_log.min()) / (coeff_log.max() - coeff_log.min() + 1e-8)

            #each ray of points distance 
            detect_y =  rotated_points[:,1] * coeff
            detect_z =  rotated_points[:,2] * coeff
            detector_points = np.stack([
                np.full_like(detect_y, -(d2-d1)),  # X = d2
                detect_y,
                detect_z
            ], axis=-1)
            point_to_detector = np.linalg.norm(detector_points - rotated_points, axis=1)
            source_p_on_source_coord = np.array([d1,0,0])
            source_to_points = np.linalg.norm(rotated_points - source_p_on_source_coord , axis=1)
            total_path_length = source_to_points + point_to_detector
            total_path_length_norm = (total_path_length - total_path_length.min()) / (total_path_length.max() - total_path_length.min())

            ratio_distance_of_points_each_ray = ( 1 - (source_to_points / total_path_length))


            #attribute_list.append(distances_norm)
            #attribute_list.append(elevation_angles_rad_norm)
            attribute_list.append(coeff_log_norm)
            attribute_list.append(ratio_distance_of_points_each_ray)
            attribute_list.append(total_path_length_norm)
            attribute_list = np.stack(attribute_list,axis=-1)
            views_spatial_attribute.append(attribute_list)
            #pdb.set_trace()
            
        #pdb.set_trace()    
        views_spatial_attribute = np.stack(views_spatial_attribute , axis=0)

        return views_spatial_attribute




    def caculate_projection_distance_v2(self, points , angle):
        d1 = self.DSO
        d2 = self.DSD 
        #pdb.set_trace()
        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        points *= self.v_res * self.v_spacing # mm

        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [            0,              0, 1]
        ])
        #pdb.set_trace()
        rotated_points = points @ rot_M.T  
        coeff = (d2) / (d1 - rotated_points[:,0])
        #pdb.set_trace()
        source_points = np.array([d1 , 0, 0 ])
        source_points = source_points @ rot_M.T
        #pdb.set_trace()
        dis_vector = rotated_points - source_points

        distances = np.linalg.norm(dis_vector,axis=1)
        #  arccos = ( (a b) / (|a| * |b|) )
        cos_angle_from_source = rotated_points[:,0] / distances
        #pdb.set_trace()
        cos_angle_from_source = np.clip(cos_angle_from_source, -1 , 1)
        angle_from_source_rad = np.arccos(cos_angle_from_source)
        distances = (distances - distances.min()) / (distances.max() - distances.min())
        #pdb.set_trace()
        #angle_from_source_deg = np.degrees(angle_from_source_rad)

        return distances , coeff , angle_from_source_rad


    def calculate_projection_distance(self ,  points , angle ):
        d1 = self.DSO
        d2 = self.DSD 

        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        points *= self.v_res * self.v_spacing # mm

        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [            0,              0, 1]
        ])
        rotated_points = points @ rot_M.T  
        pdb.set_trace()
        source_to_point = d1 - rotated_points[:, 0]  # 源点到物体点的距离
        distances = np.abs(d2 - source_to_point)  # 点到探测器平面的距离        return coeff 
        distances_ratio = distances / d2
     
        #pdb.set_trace() 
        return distances_ratio