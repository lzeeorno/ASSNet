# ------------------------------------------------------------
# Copyright (c) University of Macau and
# Shenzhen Institutes of Advanced Technology，Chinese Academy of Sciences.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# Written by FuChen Zheng(orcid:https://orcid.org/0009-0001-8589-7026)
# ------------------------------------------------------------
import numpy as np
import torch.utils.data
import argparse
from PIL import Image
from utilities.utils import mask_to_onehot

'''
150 = organ
255 = tumor
0   = background 
'''
lits_palette = [[0], [150], [255]]  # one-hot的颜色表
lits_num_classes = len(lits_palette)
'''
[150,spleen脾][255,right kidney右肾][100,left kidney左肾][200,Gallbladder胆囊]
[180,liver肝脏][220,stomach胃][130,aorta主动脉][160,pancreas胰脏]
'''
synapse_palette = [[0], [130], [200], [100], [255], [180], [160], [150], [220]]
synapse_num_classes = len(synapse_palette)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upper', default=200)
    parser.add_argument('--lower', default=-200)
    parser.add_argument('--img_size', default=512)
    parser.add_argument('--num_class', default=3)
    args = parser.parse_args()

    return args


class Dataset_ssl_lits2017_png_unlabeled(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, transform=None):
        self.args = args
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        #load npy file
        npimage = np.load(img_path, allow_pickle=True)
        #(3,512,512) -> (512,512,3)
        npimage = npimage.transpose((2, 0, 1))
        npimage = np.abs(npimage.astype("complex64"))
        # print("ct.size:{}".format(npimage.shape))
        # print("seg.size:{}".format(nplabel.shape))
        # exit()
        return npimage

class Dataset_ssl_lits2017_png(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, transform=None):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.palette = lits_palette
        self.num_classes = lits_num_classes

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        ct = Image.open(img_path)
        seg = Image.open(mask_path)
        npimage = np.array(ct)
        npmask = np.array(seg)

        # 将灰度图像的形状从 (512, 512) 增纬为 (512, 512, 1)
        # npimage = npimage[:, :, np.newaxis]
        npimage = np.expand_dims(npimage, axis=2)   #[512,512,1]
        # npimage = np.concatenate([npimage, npimage], axis=2) #[512,512,2]
        npmask = np.expand_dims(npmask, axis=2)     #[512,512,1]
        npmask = mask_to_onehot(npmask, self.palette)  #[512, 512, 3]

        npmask = npmask.transpose([2, 0, 1])
        npimage = npimage.transpose([2, 0, 1])

        npimage = npimage.astype("float32")
        npmask = npmask.astype("float32")
        # print("ct.size:{}".format(npimage.shape))
        # print("seg.size:{}".format(npmask.shape))
        # exit()

        return npimage, npmask

class Dataset_synapse_png(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, transform=None):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        '''
        [150,spleen脾][255,right kidney右肾][100,left kidney左肾][200,Gallbladder胆囊]
        [180,liver肝脏][220,stomach胃][130,aorta主动脉][160,pancreas胰脏]
        '''
        self.palette = synapse_palette
        self.num_classes = synapse_num_classes

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        ct = Image.open(img_path)
        seg = Image.open(mask_path)
        npimage = np.array(ct)
        npmask = np.array(seg)

        # 将灰度图像的形状从 (512, 512) 增纬为 (512, 512, 1)
        # npimage = npimage[:, :, np.newaxis]
        npimage = np.expand_dims(npimage, axis=2)   #[512,512,1]
        # npimage = np.concatenate([npimage, npimage], axis=2) #[512,512,2]
        npmask = np.expand_dims(npmask, axis=2)     #[512,512,1]
        npmask = mask_to_onehot(npmask, self.palette)  #[512, 512, 9]

        npmask = npmask.transpose([2, 0, 1])
        npimage = npimage.transpose([2, 0, 1])

        npimage = npimage.astype("float32")
        npmask = npmask.astype("float32")
        # print('dataset check shape')
        # print("ct.size:{}".format(npimage.shape))
        # print("seg.size:{}".format(npmask.shape))
        # exit()
        #
        return npimage, npmask
