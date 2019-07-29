import os
import numpy as np
import argparse
import argoverse
import shutil
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.camera_stats import get_image_dims_for_camera as get_dim

def match_and_return(outer_loop, inner_loop):
    return_list = []
    for i in range(len(outer_loop)):
            for j in range(len(inner_loop)):
                if(outer_loop[i] in inner_loop[j]):
                    return_list.append(inner_loop[j])
    return return_list

# the main routine
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Pseudo Ground Truth Dispairty')
    parser.add_argument('--root_dir', type=str,
                        default='/data/cmpe297-03-sp19/PilotA/Argoverse_3d_tracking/argoverse-tracking/')
    args = parser.parse_args()

    for i in range(4):
        i = i + 1
        disparity_dir = args.root_dir + 'disparity' + str(i) + '/'
        stereo_left_dir = args.root_dir + 'stereo_left' + str(i) + '/'
        stereo_right_dir = args.root_dir + 'stereo_right' + str(i) + '/'
        subroot_dir = args.root_dir + 'train' + str(i) + '/'

        argoverse_loader = ArgoverseTrackingLoader(subroot_dir)
        print('Total number of logs:',len(argoverse_loader))
        argoverse_loader.print_all()

        camL = argoverse_loader.CAMERA_LIST[7] # left stereo
        camR = argoverse_loader.CAMERA_LIST[8] # right stereo
        (width, height) = get_dim(camL)
        baseline = 0.3

        for log_id in argoverse_loader.log_list:
            ## NOTE!
            ## log ids are mixed with '-' and '_'
            ## therefore, we are going to change all '_' to '-'
            new_log_id = log_id.replace('_', '-')
            get_log_identifier = new_log_id.split('-')[0]
            print("processing log:", log_id)
            argoverse_data = argoverse_loader.get(log_id)
            calibL = argoverse_data.get_calibration(camL)
            calibR = argoverse_data.get_calibration(camR)
            focalX_px = calibL.K[0,0]
            baseline_focal = focalX_px * baseline

            if not os.path.isdir(disparity_dir):
                os.makedirs(disparity_dir)
            if not os.path.isdir(stereo_left_dir):
                os.makedirs(stereo_left_dir)
            if not os.path.isdir(stereo_right_dir):
                os.makedirs(stereo_right_dir)
            
            lidar_list = argoverse_data.lidar_list
            lidar_timestamp_list = argoverse_data.lidar_timestamp_list
            
            stereo_left_list = argoverse_data.image_list
            stereo_left_timestamp_list = argoverse_data.image_timestamp_list['stereo_front_left']
            
            # take only the first ten digits of the timestamps
            # first ten digits tell us the time by seconds
            lidar_timeonly = [str(a)[:10] for a in lidar_timestamp_list]
            stereo_timeonly = [str(a)[:10] for a in stereo_left_timestamp_list]
            
            # store the overlapping timeframes
            # overlap is the list of 10 digits timestamp that are both in stereo and lidar
            #overlap = match_and_return(stereo_timeonly, lidar_timeonly)
            
            overlap = []
            for i in range(len(stereo_timeonly)):
                for j in range(len(lidar_timeonly)):
                    if(lidar_timeonly[j] == stereo_timeonly[i]):
                        overlap.append(stereo_timeonly[i])

            # sort only corresponding lidar files (timestamp only)
            only_lidar_time = [] 
            for i in range(len(overlap)):
                for j in range(len(lidar_timestamp_list)):
                        if(overlap[i] in str(lidar_timestamp_list[j])):
                            only_lidar_time.append(lidar_timestamp_list[j])

            # sort only corresponding lidar files (absolute file location)
            only_lidar = match_and_return(overlap, lidar_list)

            # sort only corresponding stereo left files (absolute file location)
            only_left = match_and_return(overlap, stereo_left_list['stereo_front_left'])
                        
            # sort only corresponding stereo right files (absolute file location)
            only_right = match_and_return(overlap, stereo_left_list['stereo_front_right'])
                        
            for idx in range(len(only_lidar)):
                lidar_timestamp = only_lidar_time[idx]
                print("index: ", idx, "current timestamp: ", lidar_timestamp)
                
                pc = load_ply(only_lidar[idx])
                
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
                # making all negative values of disparity to -1.0
                disp_map[disp_map < 0] = -1.0
                disp_store_name = get_log_identifier + '_' + str(lidar_timestamp)
                np.save(disparity_dir + disp_store_name, disp_map)
                
            #copy corresponding left and right images to dedicated location
            for i in range(len(overlap)):
                get_left_name = only_left[i].split('/')[-1]
                get_right_name = only_right[i].split('/')[-1]

                left_store_name = get_log_identifier + '_' + get_left_name
                right_store_name = get_log_identifier + '_' + get_right_name

                shutil.copy2(only_left[i], stereo_left_dir + left_store_name)
                shutil.copy2(only_right[i], stereo_right_dir + right_store_name)