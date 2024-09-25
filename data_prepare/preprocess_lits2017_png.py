import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import cv2
import argparse
from PIL import Image
import cv2
from skimage.transform import resize
import shutil

ct_name = ".nii"
mask_name = ".nii"

ct_path = '../data/LITS2017/ct'
seg_path = '../data/LITS2017/label'
png_path = './png/'

outputImg_path = "../data/trainImage_lits2017_png"
outputMask_path = "../data/trainMask_lits2017_png"


if os.path.exists(outputImg_path):
    shutil.rmtree(outputImg_path)
    print('Clean past CT done')
    if os.path.exists(outputMask_path):
        shutil.rmtree(outputMask_path)
        print('Clean past Mask done')
if not os.path.exists(outputImg_path):
    os.mkdir(outputImg_path)
if not os.path.exists(outputMask_path):
    os.mkdir(outputMask_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    # data preprocessing
    parser.add_argument('--upper', default=200)
    parser.add_argument('--lower', default=-200)
    parser.add_argument('--tri', default=True, action='store_true')
    args = parser.parse_args()

    return args


def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files


def crop_center(img, croph, cropw):
    height, width = img[0].shape
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    return img[:, starth:starth + croph, startw:startw + cropw]


if __name__ == "__main__":
    args = parse_args()

    for index, file in enumerate(os.listdir(ct_path)):

        # read ct image
        ct_src = sitk.ReadImage(os.path.join(ct_path, file), sitk.sitkInt16)
        mask = sitk.ReadImage(os.path.join(seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        # GetArrayFromImage()turn nii into npy
        ct_array = sitk.GetArrayFromImage(ct_src)
        mask_array = sitk.GetArrayFromImage(mask)

        # gray threshold
        ct_array[ct_array > args.upper] = args.upper
        ct_array[ct_array < args.lower] = args.lower

        ct_array = ct_array.astype(np.float32)
        # ct_array = ct_array / 200
        ct_array = (ct_array - args.lower) / (args.upper - args.lower)

        # find these slices that liver start and end
        z = np.any(mask_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]
        # only take slice with liver or tumor
        start_slice = max(0, start_slice - 1)
        end_slice = min(mask_array.shape[0] - 1, end_slice + 2)
        # feature fusion
        ct_crop = ct_array[start_slice:end_slice, :, :]
        fusion_layer = 1
        mask_crop = mask_array[start_slice+fusion_layer:end_slice-fusion_layer, :, :]

        # cut whole numpy array from (512,512) -> (h,w) you need
        ct_crop = ct_crop[:, :, :]
        mask_crop = mask_crop[:, :, :]

        # unique, counts = np.unique(mask_crop, return_counts=True)
        # print(np.asarray((unique,counts)).T)
        # exit()

        print('[{}/{}]'.format(index+1, len(os.listdir(ct_path))))
        print('ct_crop.shape', ct_crop.shape)
        print('Mask_crop.shape', mask_crop.shape)
        # quit()

        # if all mask without any 0 pixel label
        if int(np.sum(mask_crop)) != 0:
            # for each slice in (n,448,448)
            for n_slice in range(mask_crop.shape[0]):
                ctImg = ct_crop[n_slice, :, :]
                maskImg = mask_crop[n_slice, :, :]
                # merge near slice into a 3 channels image for context connection
                # ctImageArray = np.zeros((ct_crop.shape[1], ct_crop.shape[2], 2 * fusion_layer + 1), np.cfloat)
                # ctImageArray[:, :, 0] = ct_crop[n_slice, :, :]
                # ctImageArray[:, :, 1] = ct_crop[n_slice + 1, :, :]
                # ctImageArray[:, :, 2] = ct_crop[n_slice + 2, :, :]
                # ctImageArray[:, :, 3] = ct_crop[n_slice + 3, :, :]
                # ctImageArray[:, :, 4] = ct_crop[n_slice + 4, :, :]
                segImg = np.zeros_like(maskImg, dtype=np.uint8)
                segImg[maskImg == 0] = 0   #background
                segImg[maskImg == 1] = 150 #liver is gray
                segImg[maskImg == 2] = 255 #tumor is white
                #add more if more labels in 1 mask (mult-task seg)
                #。。。

                #merge real and imaginary part of ct&feature fusion
                # mean merge
                # npimage = np.mean(npimage, axis=2)
                # weighted merge
                # weights = [0.2, 0.6, 0.2]
                # ctImageArray = np.dot(ctImageArray, weights)


                # ctImageArray_real = np.abs(ctImageArray)
                # ctImageArray_uint8 = (ctImageArray_real*255 / np.max(ctImageArray_real)).astype(np.uint8)

                # print('ct_crop.shape', ctImageArray_uint8.shape)
                # print('Mask_crop.shape', segImg.shape)
                #
                # print('ct_crop.shape', ctImage_save.shape)
                # print('Mask_crop.shape', segImg_save.shape)
                # exit()
                # for fit pretrained size: [448,448]->[384,384]
                ct_resized = resize(ctImg, (512, 512), anti_aliasing=True)
                # seg_resized = resize(segImg, (384, 384), anti_aliasing=True)
                ct_resized = (ct_resized*255).astype(np.uint8)
                # seg_resized = (seg_resized*255).astype(np.uint8)
                if args.tri == True:
                    args.tri = False
                    print('ct_crop.shape', ct_resized.shape)
                    print('Mask_crop.shape', segImg.shape)
                #save png
                ct_save_path = outputImg_path + "/" + str(index + 1) + "_" + str(n_slice) + ".png"
                seg_save_path = outputMask_path + "/" + str(index + 1) + "_" + str(n_slice) + ".png"

                ct = Image.fromarray(ct_resized).rotate(180, expand=False)
                seg = Image.fromarray(segImg).rotate(180, expand=False)
                ct.save(ct_save_path)
                seg.save(seg_save_path)
        args.tri = True
        print('num slice:{}'.format(mask_crop.shape[0]))

