import random

import numpy as np
import preprocess
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return np.load(path).astype(np.float32)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left) # PIL.Image (2464,2056)
        right_img = self.loader(right) # PIL.Image
        dataL = self.dploader(disp_L) # np.float32

        
        maxpool = torch.nn.MaxPool2d((4,2), stride=0, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        PIL_to_Tensor = torchvision.transforms.ToTensor()
        Tensor_to_PIL = torchvision.transforms.ToPILImage()

        maxpool_left_out = maxpool(PIL_to_Tensor(left_img)) # Tensor (514,1232)
        maxpool_right_out = maxpool(PIL_to_Tensor(right_img)) #Tensor (514,1232)

        left_img = Tensor_to_PIL(maxpool_left_out) #PIL.Image (514,1232)
        right_img = Tensor_to_PIL(maxpool_right_out) #PIL.Image(514,1232)

        print(left_img.size)
        print(right_img.size)

        print("before = " + str(dataL.shape))

        tensored_dataL = torch.from_numpy(np.expand_dims(dataL, axis=0))
        tensored_dataL_maxpool = maxpool(tensored_dataL)
        dataL = tensored_dataL_maxpool.numpy()
        dataL = np.squeeze(dataL, axis=0)
        # dataL = np.swapaxes(dataL,0,1)
        print("after = " + str(dataL.shape))
        

        #add max pooling here before crop: 
        #pooling ratio is hyper-parameter that can be tuned. so far as 8:1.


        if self.training:
            w, h = left_img.size # w = 1232, h = 514
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]
            
            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

        else:
            w, h = left_img.size

            # need to modify crop size for Argo, so that cropped pic dim is 
            # divisable by 32.

            # left_img = left_img.crop((w - 1232, h - 368, w, h))
            # right_img = right_img.crop((w - 1232, h - 368, w, h))
            left_img = left_img.crop((w - 1200, h - 352, w, h))
            right_img = right_img.crop((w - 1200, h - 352, w, h))
            w1, h1 = left_img.size

            # dataL1 = dataL[h - 368:h, w - 1232:w]
            dataL = dataL[h - 352:h, w - 1200:w]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

        dataL = torch.from_numpy(dataL).float()
        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)