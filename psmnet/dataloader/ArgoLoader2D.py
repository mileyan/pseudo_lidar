import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
# /home/cmpe/PilotA/PSMNet/dataset/argoverse-tracking/sample/

  left_fold  = 'stereo_front_left/'
  right_fold = 'stereo_front_right/'


  image_l = [img for img in os.listdir(filepath+left_fold) if img.find('stereo_front_left') > -1]
  image_r = [img for img in os.listdir(filepath+right_fold) if img.find('stereo_front_right') > -1]


  left_test  = [filepath+left_fold+img for img in image_l]
  right_test = [filepath+right_fold+img for img in image_r]

  return left_test, right_test
