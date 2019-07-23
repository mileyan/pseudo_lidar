import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    left_fold  = 'stereo_front_left/'
    right_fold = 'stereo_front_right/'
    disp_fold = 'disparity/'

    image_l = [img for img in os.listdir(filepath+left_fold) if img.find('stereo_front_left') > -1]
    image_r = [img for img in os.listdir(filepath+right_fold) if img.find('stereo_front_right') > -1]
    disp = [npy for npy in os.listdir(filepath+disp_fold) if npy.find('3159') > -1]

    left_train  = [filepath+left_fold+img for img in image_l]
    right_train = [filepath+right_fold+img for img in image_r]
    disp_train = [filepath+disp_fold+npy for npy in disp]

    return left_train, right_train, disp_train
