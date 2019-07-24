<<<<<<< HEAD
import os
import argparse
import numpy as np
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.calibration import Calibration

baseline = 0.3

def project_disp_to_depth(calib, disp, max_high):
    disp[disp < 0] = 0
    mask = disp > 0
    depth = calib.K[0,0] * baseline / (disp + 1. - mask)#invalid disparity --> depth is infinity
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])#u, v, depth
    points = points.reshape((3, -1)).T
    uv_depth = points[mask.reshape(-1)]# valid points only
    cloud = calib.project_image_to_ego(uv_depth)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

    
# the main routine
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Lidar')
    parser.add_argument('--root_dir', type=str,
                        default='/Users/mengsli/Downloads/DLRepo/argoverse-tracking/')
    parser.add_argument('--sub_folder', type=str, default='sample/') #train1, train2 ... val, test
    parser.add_argument('--max_high', type=int, default=1)
    args = parser.parse_args()

    assert os.path.isdir(args.root_dir)
    sub_dir = args.root_dir + '/' + args.sub_folder
    
    argoverse_loader = ArgoverseTrackingLoader(sub_dir)

    camL = argoverse_loader.CAMERA_LIST[7] # left stereo
    camR = argoverse_loader.CAMERA_LIST[8] # right stereo
    
    for log_id in argoverse_loader.log_list:
        argoverse_data = argoverse_loader.get(log_id)
        calibL = argoverse_data.get_calibration(camL)
        calibR = argoverse_data.get_calibration(camR)


        disparity_dir = sub_dir + '/' + log_id + '/' + 'pred_disparity/'
        assert os.path.isdir(disparity_dir)
        
        disps = [x for x in os.listdir(disparity_dir) if x[-3:] == 'npy']
        disps = sorted(disps)
        
        pred_lidar_dir = sub_dir + '/' + log_id + '/' + 'pred_lidar'
        if not os.path.isdir(pred_lidar_dir):
            os.makedirs(pred_lidar_dir)
    
        for fn in disps:
            predix = fn[:-4]    
            disp = np.load(disparity_dir + '/' + fn) #2056x2464

            lidar = project_disp_to_depth(calibL, disp, args.max_high) #nx3   

            # pad 1 in the indensity dimension
            lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1) #nx4
            lidar = lidar.astype(np.float32)
            lidar.tofile('{}/{}.ply'.format(pred_lidar_dir, 'Pseudo_PC_' + predix))
            print('Finish Depth {}'.format(predix))  

