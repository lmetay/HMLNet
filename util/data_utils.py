"""This module contains data read/save functions """
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt  

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png','.tif', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result

class MyDataset(Dataset):
    def __init__(self, args, A_path, B_path, lab_path_ori, lab_1_2_path, lab_1_4_path, lab_1_8_path):
        super(MyDataset, self).__init__()
        # 获取图片列表
        datalist = [name for name in os.listdir(A_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]
        datalist_lab_ori = [name for name in os.listdir(lab_path_ori) for item in args.suffix if
                      os.path.splitext(name)[1] == item]
        datalist_lab_1_2 = [name for name in os.listdir(lab_1_2_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]
        datalist_lab_1_4 = [name for name in os.listdir(lab_1_4_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]
        datalist_lab_1_8 = [name for name in os.listdir(lab_1_8_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]
            

        self.A_filenames = [os.path.join(A_path, x) for x in datalist if is_image_file(x)]
        self.B_filenames = [os.path.join(B_path, x) for x in datalist if is_image_file(x)]
        self.lab_ori_filenames = [os.path.join(lab_path_ori, x) for x in datalist_lab_ori if is_image_file(x)]  #for binary class
        self.lab_1_2_filenames = [os.path.join(lab_1_2_path, x) for x in datalist_lab_1_2 if is_image_file(x)]
        self.lab_1_4_filenames = [os.path.join(lab_1_4_path, x) for x in datalist_lab_1_4 if is_image_file(x)]
        self.lab_1_8_filenames = [os.path.join(lab_1_8_path, x) for x in datalist_lab_1_8 if is_image_file(x)]    


        self.transform_RGB_A = get_transform(convert=True, normalize=True, is_pre=True, isRGB=True)
        self.transform_RGB_B = get_transform(convert=True, normalize=True, is_pre=False, isRGB=True) 
        self.transform_label = get_transform() #for binary class, only convert to tensor
        self.out_cls = 2

    def __getitem__(self,index):
        fn = self.A_filenames[index]
        A_img = self.transform_RGB_A(Image.open(self.A_filenames[index]).convert('RGB')) 
        B_img = self.transform_RGB_B(Image.open(self.B_filenames[index]).convert('RGB'))

        label_ori = self.transform_label(Image.open(self.lab_ori_filenames[index]))

        label_1_2 = self.transform_label(Image.open(self.lab_1_2_filenames[index]))

        label_1_4 = self.transform_label(Image.open(self.lab_1_4_filenames[index]))

        label_1_8 = self.transform_label(Image.open(self.lab_1_8_filenames[index]))

        label = [label_ori, label_1_2, label_1_4, label_1_8]

        return A_img, B_img, label, index

    def __len__(self):
        return len(self.A_filenames)


def get_transform(convert=True, normalize=False, is_pre=True, isRGB=True):
    transform_list = [] 
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        if is_pre:
            if isRGB:
                transform_list += [transforms.Normalize((0.4355, 0.4382, 0.4335),
                                                        (0.2096, 0.2089, 0.2095))]    # LEVIR
                # transform_list += [transforms.Normalize((0.4310, 0.4498, 0.4480),
                #                                         (0.2085, 0.2069, 0.2054))]    # SYSU  
                                               
        else:
            if isRGB:
                transform_list += [transforms.Normalize((0.3306, 0.3351, 0.3297), 
                                                        (0.1612, 0.1667, 0.1607))]     # LEVIR
                # transform_list += [transforms.Normalize((0.4310, 0.4364, 0.4275), 
                #                                         (0.1966, 0.1983, 0.1934))]     # SYSU

                                
    return transforms.Compose(transform_list)



    
