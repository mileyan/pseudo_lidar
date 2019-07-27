from __future__ import print_function
import argparse
import os
import random
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess 
from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--datapath', default='/data/cmpe297-03-sp19/PilotA/Argoverse_3d_tracking/argoverse-tracking/',
                    help='select model')
parser.add_argument('--sub_folder', type=str, default='sample/') #train1, train2 ... val, test
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print("cuda available")   
    
torch.manual_seed(args.seed)
processed = preprocess.get_transform(augment=False)

## load the model ##
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def test(imgL,imgR):
    model.eval()
    if args.cuda:
        imgL = torch.FloatTensor(imgL).cuda()
        imgR = torch.FloatTensor(imgR).cuda() 
        imgL, imgR= Variable(imgL), Variable(imgR)
        with torch.no_grad():
            output = model(imgL,imgR)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()

        return pred_disp
            
def main():
    assert os.path.isdir(args.datapath)
    sub_dir = args.datapath+ '/' + args.sub_folder
    log_list = [log for log in os.listdir(sub_dir) if log is not None ] 
    for log_id in log_list:
        left_dir = sub_dir + '/' + log_id + '/' + 'stereo_front_left/'
        right_dir = sub_dir + '/' + log_id + '/' + 'stereo_front_lright/'
        assert os.path.isdir(left_dir)
        assert os.path.isdir(right_dir)
        
        test_left_img = [x for x in os.listdir(left_dir) if x[-3:] == 'jpg']
        test_right_img = [x for x in os.listdir(right_dir) if x[-3:] == 'jpg']
        test_left_img = sorted(test_left_img)
        test_right_img = sorted(test_right_img)
        
        pred_disp_dir = sub_dir + '/' + log_id + '/' + 'pred_disparity/' # to store 
        if not os.path.isdir(pred_disp_dir):
            os.makedirs(pred_disp_dir)
    
        for inx in range(len(test_left_img)):
            if (test_left_img[inx][18:28] != test_right_img[inx][18:28]): # time not equal in seconds!
                print('time not synced: ', test_left_img[inx][18:28], '!=', test_right_img[inx][18:28])
                return -1
                
            imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
            imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))
            #print('after readin: ', imgL_o.shape)

            #imgL_o = skimage.transform.resize(imgL_o, (imgL_o.shape[0] / 2, \
            #                imgL_o.shape[1] / 4), anti_aliasing=True).astype('float32')
            #imgR_o = skimage.transform.resize(imgR_o, (imgR_o.shape[0] / 2, \
            #                imgR_o.shape[1] / 4), anti_aliasing=True).astype('float32')
            
            #replace the downsampling with downscaling_local_mean
            imgL_o = skimage.transform.downscale_local_mean(imgL_o, (4,4,1))
            imgR_o = skimage.transform.downscale_local_mean(imgR_o, (4,4,1))
            #print('after downscale: ', imgL_o.shape)

            imgL = processed(imgL_o).numpy()
            imgR = processed(imgR_o).numpy()
            #print('after precessed: ', imgL.shape)

            imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
            imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])
            #print('after reshape: ', imgL.shape)

            # pad to (1056 , 640) for /2, /4 resizing
            # pad to (544 , 640) for /4, /4 rescaling
            top_pad = 544-imgL.shape[2]
            right_pad = 640-imgL.shape[3]

            imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,right_pad)),mode='constant',constant_values=0)
            imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,right_pad)),mode='constant',constant_values=0)
            #print('after padding: ', imgL.shape)

            start_time = time.time()
            pred_disp = test(imgL,imgR)
            #print('shape of predicted disparity: ',pred_disp.shape)
            print('time = %.2f' %(time.time() - start_time))

            pred_disp = pred_disp[top_pad:,:-right_pad]
            np.save(pred_disp_dir + '/pred_disp_' + test_left_img[inx][18:-3], pred_disp)
            #skimage.io.imsave(test_left_img[inx].split('/')[-1],(pred_disp*256).astype('uint16'))
            #skimage.io.imsave(test_left_img[inx].split('/')[-1],(pred_disp*256).astype('uint16'))

if __name__ == '__main__':
   main()






