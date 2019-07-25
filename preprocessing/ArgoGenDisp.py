import json
import os
import numpy as np
import matplotlib.pyplot as plt
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.calibration import Calibration, determine_valid_cam_coords
from argoverse.utils.camera_stats import get_image_dims_for_camera as get_dim

root_dir =  '/data/cmpe297-03-sp19/PilotA/argoverse-api/argoverse-tracking/'
#subroot_dir = root_dir + 'val/'
subroot_dir = root_dir + 'sample/'
argoverse_loader = ArgoverseTrackingLoader(subroot_dir)

print('Total number of logs:',len(argoverse_loader))
argoverse_loader.print_all()
print(argoverse_loader.log_list)

camL = argoverse_loader.CAMERA_LIST[7] # left stereo
camR = argoverse_loader.CAMERA_LIST[8] # right stereo
(width, height) = get_dim(camL)
baseline = 0.3

for log_id in argoverse_loader.log_list:
    print("processing log:", log_id)
    argoverse_data = argoverse_loader.get(log_id)
    calibL = argoverse_data.get_calibration(camL)
    calibR = argoverse_data.get_calibration(camR)
    focalX_px = calibL.K[0,0]
    baseline_focal = focalX_px * baseline
    disparity_dir = subroot_dir + '/' + log_id + '/disparity/'
    if not os.path.isdir(disparity_dir):
        os.makedirs(disparity_dir)

    lidar_list = argoverse_data.lidar_list
    lidar_timestamp_list = argoverse_data.lidar_timestamp_list
    lidar_5hz_list = lidar_list[::2] # downsample to sync with stereo camera, same initial time frame
    lidar_timestamp_5hz_list = lidar_timestamp_list[::2]  	
    
    for idx in range(len(lidar_5hz_list)):
        timestamp = lidar_timestamp_5hz_list[idx]
        print("index: ", idx, "current timestamp: ", timestamp)
        pc = load_ply(lidar_5hz_list[idx])

        uv = calibL.project_ego_to_image(pc)

        fov_inds = (uv[:, 0] < width - 1) & (uv[:, 0] >= 0) & \
                   (uv[:, 1] < height - 1)& (uv[:, 1] >= 0)
        fov_inds = fov_inds & (pc[:, 0] > 1) # filters out points that are behind camera
    
        valid_uv = uv[fov_inds, :]  
        valid_uv = np.round(valid_uv).astype(int)
        
        valid_pc = pc[fov_inds, :]        
        valid_uvd = calibL.project_ego_to_cam(valid_pc)

        depth_map = np.zeros((height, width)) - 1

        for i in range(valid_uv.shape[0]):
            depth_map[int(valid_uv[i, 1]), int(valid_uv[i, 0])] = valid_uvd[i, 2]

        disp_map = (calibL.K[0,0] * baseline) / depth_map 
        np.save(disparity_dir + '/' + str(timestamp), disp_map)
