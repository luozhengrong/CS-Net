import numpy as np
from PIL import Image
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from random import randint
from torch.utils.data.dataset import Dataset
from pre_processing import *
from mean_std import *
import math
import copy
import cv2
from skimage import measure

def cal_crop_num_img(img_size, in_size):
    if img_size[0] % in_size[0] == 0:
        crop_n1 = math.ceil(img_size[0] / in_size[0]) + 1
    else:
        crop_n1 = math.ceil(img_size[0] / in_size[0])

    if img_size[1] % in_size[1] == 0:
        crop_n2 = math.ceil(img_size[1] / in_size[1]) + 1
    else:
        crop_n2 = math.ceil(img_size[1] / in_size[1])
    return crop_n1,crop_n2

class SEMDataTrain(Dataset):

    def __init__(self, image_path, mask_path,in_size=[512,512], out_size=[512,512],pad=0,image_num=1,nor=2):

        self.mask_arr = glob.glob(str(mask_path) + "/*")
        self.image_arr = glob.glob(str(image_path) + str("/*"))
        self.in_size, self.out_size = in_size, out_size
        self.image_path=image_path
        self.mask_path=mask_path
        self.nor=nor
        self.data_len = len(self.mask_arr)
        single_image_name = self.image_arr[0]
        img_as_img = Image.open(single_image_name)
        img_as_np = np.asarray(img_as_img)
        self.pad_size=pad
        self.image_num=image_num
        self.img_height, self.img_width = img_as_np.shape[0], img_as_np.shape[1]
        del img_as_np


    def __getitem__(self, index):

        tuple_all=[]
        flip_num = randint(0, 3)
        rotate_num = randint(0, 3)
        y_loc, x_loc = randint(0, self.img_height-self.out_size[0]+self.pad_size*2), randint(0, self.img_width-self.out_size[1]+self.pad_size*2)
        if rotate_num==0 or rotate_num==1:
            y_loc, x_loc = randint(0, self.img_width - self.out_size[0] + self.pad_size * 2), randint(0,self.img_height- self.out_size[1] + self.pad_size * 2)

        if self.image_num==1:
            num = [index]
        elif self.image_num==3:
            if index==0:
                num = [index,index,index + 1]
            elif index == (self.data_len - 1):
                num = [index - 1, index, index]
            else:
                num=[index-1,index,index+1]
        elif self.image_num==5:
            if index==0:
                num = [index, index,index, index + 1,index+2]
            elif index==1:
                num = [index-1 ,index-1, index, index+1,index+2]
            elif index==(self.data_len-2):
                num = [index-2,index-1 , index, index+1,index+1]
            elif index == (self.data_len - 1):
                num = [index-2,index - 1, index, index,index]
            else:
                num=[index-2,index-1,index,index+1,index+2]
        elif self.image_num==7:
            if index==0:
                num = [index ,index, index,index, index + 1,index+2,index+3]
            elif index==1:
                num = [index-1,index-1 ,index-1, index, index+1,index+2,index+3]
            elif index == 2:
                num = [index - 2, index - 2,index-1, index, index + 1, index + 2,index+3]
            elif index==(self.data_len-3):
                num = [index-3,index-2,index-1 , index, index+1,index+2,index+2]
            elif index==(self.data_len-2):
                num = [index-3,index-2,index-1 , index, index+1,index+1,index+1]
            elif index == (self.data_len - 1):
                num = [index-3,index-2,index - 1, index, index,index,index]
            else:
                num=[index-3,index-2,index-1,index,index+1,index+2,index+3]

        img_as_tensors = torch.Tensor([])
        for index_ in num:
            single_image_name = self.image_path+"train"+str(index_+1).zfill(3)+".png"
            #print(single_image_name)
            img_as_img = Image.open(single_image_name)
            img_as_img = rotate(img_as_img, rotate_num)
            img_as_np = np.asarray(img_as_img)

            # Augmentation
            # flip {0: vertical, 1: horizontal, 2: both, 3: none}
            img_as_np = flip(img_as_np, flip_num)
            # Noise Determine {0: Gaussian_noise, 1: uniform_noise
            if randint(0, 1):
                # Gaussian_noise
                gaus_sd, gaus_mean = randint(0, 20), 0
                img_as_np = add_gaussian_noise(img_as_np, gaus_mean, gaus_sd)
            else:
                # uniform_noise
                l_bound, u_bound = randint(-20, 0), randint(0, 20)
                img_as_np = add_uniform_noise(img_as_np, l_bound, u_bound)
            # Brightness
            pix_add = randint(-20, 20)
            img_as_np = change_brightness(img_as_np, pix_add)
            # Crop the image
            img_as_np = np.pad(img_as_np, self.pad_size, mode="symmetric")
            img_as_np = cropping(img_as_np, crop_size=self.in_size, dim1=y_loc, dim2=x_loc)
            # Normalize the image
            if img_as_np.max()==0 and img_as_np.min()==0:
                pass
            elif self.nor==2:
                img_as_np = normalization2(img_as_np,1,0)
            elif self.nor==1:
                img_as_np = normalization1(img_as_np)
            img_as_np = np.expand_dims(img_as_np, axis=0)  # add additional dimension
            img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor
            img_as_tensors=torch.cat((img_as_tensors,img_as_tensor),dim=0)

        tuple_all.append(img_as_tensors)
        """
        # GET MASK
        """
        single_mask_name0 = self.mask_path + "/train" + str(num[1]+1 ).zfill(3) + ".png"
        single_mask_name1 =self.mask_path+ "/train"+str(num[2]+1).zfill(3)+".png"
        single_mask_name2 = self.mask_path + "/train" + str(num[3]+1).zfill(3) + ".png"
        label_adress = [single_mask_name0,single_mask_name1,single_mask_name2]
        for n,adress in enumerate(label_adress):
            msk_as_img = Image.open(adress)
            msk_as_img = rotate(msk_as_img, rotate_num)
            msk_as_np = np.asarray(msk_as_img)
            msk_as_np = flip(msk_as_np, flip_num)
            msk_as_np = np.pad(msk_as_np, self.pad_size, mode="symmetric")
            msk_as_np = cropping(msk_as_np, crop_size=self.out_size, dim1=y_loc, dim2=x_loc)
            msk_as_np = msk_as_np // 255

            #×××××××××××××××××××××××××××××××××
            if n ==1:
                msk_as_np2 = np.pad(msk_as_np, 64, mode="constant")
                dist = cv2.distanceTransform(src=msk_as_np2, distanceType=cv2.DIST_L2, maskSize=3)
                dist = cv2.normalize(dist, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                dist = dist[64:-64, 64:-64]
                dist = 255 - dist
                dist[dist == 255] = 0
                dist = dist.astype(float)
                dist = dist / 255
                dist = torch.from_numpy(dist).float()
                tuple_all.append(dist)

            #********************************
            weight = np.ones(shape=msk_as_np.shape, dtype=np.float32)
            contours = measure.label(msk_as_np)
            max = 0
            for m in range(1, contours.max() + 1):
                s = np.sum(contours == m)
                # print(s)
                if s <= 400:
                    s = 400
                if s > max:
                    max = s
                weight[contours == m] = 1 / pow(np.log(s), 2)
            weight[contours == 0] = 1 / pow(np.log(max), 2)

            msk_as_weight2 = torch.from_numpy(weight).float()
            msk_as_weight1 = np.expand_dims(msk_as_weight2, axis=0)

            msk_as_tensor = torch.from_numpy(msk_as_np).long()
            tuple_all.append(msk_as_tensor)
            tuple_all.append(msk_as_weight1)
        tuple_all = tuple(tuple_all)
        return tuple_all

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len


class SEMDataVal(Dataset):
    def __init__(self, image_path, mask_path, in_size=[512,512], out_size=[512,512],pad=0,image_num=1,nor=2):
        '''
        Args:
            image_path = path where test images are located
            mask_path = path where test masks are located
        '''
        # paths to all images and masks
        self.image_path=image_path
        self.mask_path=mask_path
        self.mask_arr = glob.glob(str(mask_path) + str("/*"))
        self.image_arr = glob.glob(str(image_path) + str("/*"))
        self.in_size = in_size
        self.out_size = out_size
        self.image_num=image_num
        self.nor=nor
        self.data_len = len(self.mask_arr)
        self.pad=pad
    def __getitem__(self, index):

        pad_size=self.pad
        img_as_tensors = torch.Tensor([])
        single_image = self.image_arr[index]

        name = str(index)

        if self.image_num==1:
            num = [index]
        elif self.image_num==3:
            if index==0:
                num = [index,index,index + 1]
            elif index == (self.data_len - 1):
                num = [index - 1, index, index]
            else:
                num=[index-1,index,index+1]
        elif self.image_num==5:
            if index==0:
                num = [index, index,index, index + 1,index+2]
            elif index==1:
                num = [index-1 ,index-1, index, index+1,index+2]
            elif index==(self.data_len-2):
                num = [index-2,index-1 , index, index+1,index+1]
            elif index == (self.data_len - 1):
                num = [index-2,index - 1, index, index,index]
            else:
                num=[index-2,index-1,index,index+1,index+2]
        elif self.image_num==7:
            if index==0:
                num = [index ,index, index,index, index + 1,index+2,index+3]
            elif index==1:
                num = [index-1,index-1 ,index-1, index, index+1,index+2,index+3]
            elif index == 2:
                num = [index - 2, index - 2,index-1, index, index + 1, index + 2,index+3]
            elif index==(self.data_len-3):
                num = [index-3,index-2,index-1 , index, index+1,index+2,index+2]
            elif index==(self.data_len-2):
                num = [index-3,index-2,index-1 , index, index+1,index+1,index+1]
            elif index == (self.data_len - 1):
                num = [index-3,index-2,index - 1, index, index,index,index]
            else:
                num=[index-3,index-2,index-1,index,index+1,index+2,index+3]

        for index_ in num:
            single_image = self.image_path+"test"+str(index_+1).zfill(3)+".png"
            img_as_img = Image.open(single_image)
            img_as_np = np.asarray(img_as_img)
            img_as_np = np.pad(img_as_np, pad_size, mode="symmetric")
            img_as_np = np.expand_dims(img_as_np, axis=0)
            if img_as_np.max() == 0 and img_as_np.min() == 0:
                pass
            elif self.nor == 2:
                img_as_np = normalization2(img_as_np, 1, 0)
            elif self.nor == 1:
                img_as_np = normalization1(img_as_np)
            img_as_tensor = torch.from_numpy(img_as_np).float()

            img_as_tensors = torch.cat((img_as_tensors, img_as_tensor), dim=0)

        return (img_as_tensors,int(index))
    def __len__(self):
        return self.data_len

class SEMDataTest(Dataset):
    def __init__(self, image_path, mask_path, in_size=[512,512], out_size=[512,512],pad=0,image_num=5,nor=2):

        self.image_path=image_path
        self.mask_path=mask_path
        self.mask_arr = glob.glob(str(mask_path) + str("/*"))
        self.image_arr = glob.glob(str(image_path) + str("/*"))
        self.in_size = in_size
        self.out_size = out_size
        self.image_num=image_num
        self.nor=nor
        self.data_len = len(self.mask_arr)
        self.pad=pad
    def __getitem__(self, index):

        pad_size=self.pad
        img_as_tensors = torch.Tensor([])
        single_image = self.image_arr[index]

        name = str(index)
        if self.image_num==1:
            num = [index]
        elif self.image_num==3:
            if index==0:
                num = [index,index,index + 1]
            elif index == (self.data_len - 1):
                num = [index - 1, index, index]
            else:
                num=[index-1,index,index+1]
        elif self.image_num==5:
            if index==0:
                num = [index, index,index, index + 1,index+2]
            elif index==1:
                num = [index-1 ,index-1, index, index+1,index+2]
            elif index==(self.data_len-2):
                num = [index-2,index-1 , index, index+1,index+1]
            elif index == (self.data_len - 1):
                num = [index-2,index - 1, index, index,index]
            else:
                num=[index-2,index-1,index,index+1,index+2]
        elif self.image_num==7:
            if index==0:
                num = [index ,index, index,index, index + 1,index+2,index+3]
            elif index==1:
                num = [index-1,index-1 ,index-1, index, index+1,index+2,index+3]
            elif index == 2:
                num = [index - 2, index - 2,index-1, index, index + 1, index + 2,index+3]
            elif index==(self.data_len-3):
                num = [index-3,index-2,index-1 , index, index+1,index+2,index+2]
            elif index==(self.data_len-2):
                num = [index-3,index-2,index-1 , index, index+1,index+1,index+1]
            elif index == (self.data_len - 1):
                num = [index-3,index-2,index - 1, index, index,index,index]
            else:
                num=[index-3,index-2,index-1,index,index+1,index+2,index+3]

        for index_ in num:
            single_image = self.image_path+"test"+str(index_+1).zfill(3)+".png"
            img_as_img = Image.open(single_image)
            img_as_np = np.asarray(img_as_img)
            # clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            # img_as_np = clahe.apply(img_as_np)
            # img_as_np = np.power(img_as_np / float(np.max(img_as_np)), 1.2)
            img_as_np = np.pad(img_as_np, pad_size, mode="symmetric")
            img_as_np = np.expand_dims(img_as_np, axis=0)
            if img_as_np.max() == 0 and img_as_np.min() == 0:
                continue
            elif self.nor == 2:
                img_as_np = normalization2(img_as_np, 1, 0)
            elif self.nor == 1:
                img_as_np = normalization1(img_as_np)
            img_as_tensor = torch.from_numpy(img_as_np).float()

            img_as_tensors = torch.cat((img_as_tensors, img_as_tensor), dim=0)


        return (img_as_tensors,int(index))
    def __len__(self):
        return self.data_len

if __name__ == "__main__":

    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    SEM_test = SEMDataTest(
        '../data/test/images/', '../data/test/masks')
    SEM_val = SEMDataVal('../data/val/images', '../data/val/masks')

    imag_1, msk = SEM_val.__getitem__(0)
    print(imag_1.shape)