import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import cv2
import argparse
from skimage.transform import resize

ct_name = ".nii"
mask_name = ".nii"

ct_path = '../data/LITS2017/ct'
seg_path = '../data/LITS2017/label'
png_path = './png/'

outputImg_path = "../data/trainImage_lits2017_npy"
outputMask_path = "../data/trainMask_lits2017_npy"

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
    parser.add_argument('--tri', default=True, help='whether to print shape')
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

        # mask_array[mask_array == 1] = 0  # 肿瘤
        # mask_array[mask_array == 2] = 1

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
        # cut horizontily, only save slice carry liver
        ct_crop = ct_array[start_slice:end_slice, :, :]
        fusion_layer = 1
        mask_crop = mask_array[start_slice + fusion_layer:end_slice - fusion_layer, :, :]

        # resize whole numpy array  (512,512)->(h,w) you want
        ct_crop = ct_crop[:, :, :]
        mask_crop = mask_crop[:, :, :]

        print('[{}/{}]'.format(index+1, len(os.listdir(ct_path))))
        # print('ct_crop.shape', ct_crop.shape)

        # if all mask without any 0 pixel label
        if int(np.sum(mask_crop)) != 0:
            # for each slice in (n,448,448)
            for n_slice in range(mask_crop.shape[0]):
                # merge near slice into a 3 channels image for context connection
                ctImageArray = np.zeros((512, 512, 2*fusion_layer+1), np.cfloat)
                ctImageArray[:, :, 0] = ct_crop[n_slice, :, :]
                ctImageArray[:, :, 1] = ct_crop[n_slice + 1, :, :]
                ctImageArray[:, :, 2] = ct_crop[n_slice + 2, :, :]
                # ctImageArray[:, :, 0] = resize(ct_crop[n_slice, :, :], (512, 512), anti_aliasing=True)
                # ctImageArray[:, :, 1] = resize(ct_crop[n_slice + 1, :, :], (512, 512), anti_aliasing=True)
                # ctImageArray[:, :, 2] = resize(ct_crop[n_slice + 2, :, :], (512, 512), anti_aliasing=True)

                # for fit pretrained size: [448,448]->[384,384]
                # maskImg = resize(mask_crop[n_slice, :, :], (512, 512), anti_aliasing=True)
                maskImg = mask_crop[n_slice, :, :]
                if args.tri == True:
                    args.tri = False
                    print('ct_crop.shape', ctImageArray.shape)
                    print('Mask_crop.shape', maskImg.shape)
                # exit()
                imagename = outputImg_path + "/" + str(index + 1) + "_" + str(n_slice) + ".npy"
                maskname = outputMask_path + "/" + str(index + 1) + "_" + str(n_slice) + ".npy"

                np.save(imagename, ctImageArray)  # (512，512,3) np.float dtype('float64')
                np.save(maskname, maskImg)  #dtype('uint8') value is 0 1 2
        else:
            continue
        print("num_slice:{}".format(mask_crop.shape[0]))
        args.tri = True

