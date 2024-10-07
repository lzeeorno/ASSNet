# -*- coding: utf-8 -*-

import os
import argparse
from glob import glob
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split

from utilities.metrics import dice_coef_lits, iou_score, fnr_score, fpr_score, assd_score, rmsd_score, msd_score, accuracy,f1_score, ppv, sensitivity, rvd_score
import utilities.losses as loss_fun
from torch.utils.data import DataLoader
from dataset.dataset import Dataset_ssl_lits2017_png
from net.ASSNet import ASSNet
from calflops import calculate_flops
import pandas as pd
from PIL import Image
import numpy as np




# test_ct_path = '../data/testImage'   #需要预测的CT图像
# test_seg_path = '../data/testMask' #需要预测的CT图像标签，如果要在线提交codelab，需要先得到预测过的70例肝脏标签


'''
mandatory change 
if model change
'''
model_path = '../trained_models/LiTS2017_ASSNet/2024-10-07-17-07-54/epoch0-val_loss:0.6414-val_dice2:0.3569_model.pth'
csv_path = '../trained_models/LiTS2017_ASSNet/2024-10-07-17-07-54'
save_pred_png_trigger = True
pred_png_path = '../pred_result/LiTS2017_ASSNet/'
if not os.path.exists(pred_png_path):
    os.makedirs(pred_png_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=None,
                        help='')
    parser.add_argument('--training', type=bool, default=False,
                    help='whthere dropout or not')
    # dataset name on log record
    parser.add_argument('--dataset', default='LiTS2017',
                        help='default: LiTS2017')
    # mode name on log record
    parser.add_argument('--model_name', default='ASSNet',
                        choices=['Unet', 'res_unet_plus', 'R2Unet', 'sepnet', 'KiU_Net'])
    parser.add_argument('-b', '--batch_size', default=10, type=int,
                        metavar='N', help='calculate FlOPs batch size(default: 16), R2U=6, ResUnet+= 16')
    args = parser.parse_args()

    return args

'''
mandatory change 
if model change
'''



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    args = parse_args()

    # Data loading
    img_paths = sorted(glob('../data/testImage_lits2017_png/*'))
    mask_paths = sorted(glob('../data/testMask_lits2017_png/*'))
    # test_img_paths, giveup_ct, test_mask_paths, giveup_mask = \
    #     train_test_split(img_paths, mask_paths, test_size=0.1, random_state=2024)

    test_dataset = Dataset_ssl_lits2017_png(args, img_paths, mask_paths,None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    # print("test_num:%s" % str(len(test_img_paths)))
    # print("give_up num:%s" % str(len(giveup_ct)))

    # create model
    print("=> creating model %s" % args.model_name)
    #choices = ['Unet', 'res_unet_plus', 'R2Unet', 'sepnet', 'KiU_Net']
    if args.model_name == 'Unet':
        model = Unet.U_Net(args)
    if args.model_name == 'res_unet_plus':
        model = resunet_pp.ResUnetPlusPlus(args)
    if args.model_name == 'KiU_Net':
        model = KiU_Net.kiunet_org(args)
    if args.model_name == 'R2Unet':
        model = R2Unet.R2U_Net(args)
    if args.model_name == 'sepnet':
        model = sepnet.sepnet(args)
    if args.model_name == 'ASSNet':
        model = ASSNet(args)

    model = torch.nn.DataParallel(model).cuda()

    model.load_state_dict(
        torch.load(model_path)
    )
    model.eval()
    #calculate Flops
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=(args.batch_size, 1,512,512),
                                          output_as_string=True,
                                          print_results=True,
                                          print_detailed=True,
                                          output_unit='M'
                                          )
    print('%s -- FLOPs:%s  -- MACs:%s   -- Params:%s \n'%(args.model_name,flops, macs, params))
    print("test_num:%s" % str(len(img_paths)))

    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()
    voe_s = AverageMeter()
    fnr_s = AverageMeter()
    fpr_s = AverageMeter()
    assd_s = AverageMeter()
    rmsd_s = AverageMeter()
    msd_s = AverageMeter()
    acc_s = AverageMeter()
    f1_s = AverageMeter()
    Precision_s = AverageMeter()
    Recall_s = AverageMeter()
    RVD_s = AverageMeter()


    criterion = loss_fun.BCEDiceLoss_lits2017().cuda()
    cudnn.benchmark = True

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):

            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)

            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice_1 = dice_coef_lits(output, target)[0]
            dice_2 = dice_coef_lits(output, target)[1]
            voe = 1 - iou
            rvd = rvd_score(output, target)
            fnr = fpr_score(output, target)
            fpr = fpr_score(output, target)
            assd = assd_score(output, target)
            rmsd = rmsd_score(output, target)
            msd = msd_score(output, target)
            acc = accuracy(output, target)
            f1 = f1_score(output, target)
            Precision = ppv(output, target)
            Recall = sensitivity(output, target)



            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices_1s.update(torch.tensor(dice_1), input.size(0))
            dices_2s.update(torch.tensor(dice_2), input.size(0))
            voe_s.update(voe, input.size(0))
            RVD_s.update(rvd, input.size(0))
            fnr_s.update(fnr, input.size(0))
            fpr_s.update(fpr, input.size(0))
            assd_s.update(assd, input.size(0))
            rmsd_s.update(rmsd, input.size(0))
            msd_s.update(msd, input.size(0))
            acc_s.update(acc, input.size(0))
            f1_s.update(f1, input.size(0))
            Precision_s.update(Precision, input.size(0))
            Recall_s.update(Recall, input.size(0))

            '''
            save pred png, need large memory!!!
            '''
            if save_pred_png_trigger == True:
                for idx in range(output.shape[0]):
                    pred_img = torch.sigmoid(output[idx]).cpu().detach().numpy()#convert tensor2array
                    # output_image_combined = np.maximum(pred_img[0], pred_img[1],pred_img[2])  # 将两个通道的像素值取最大值作为合并后的图像
                    # pred_img = (output_image_combined*255).astype('uint8')

                    background = (pred_img[0, :, :] * 0).astype(np.uint8)
                    channel1 = (pred_img[1, :, :] * 150).astype(np.uint8)
                    channel2 = (pred_img[2, :, :] * 255).astype(np.uint8)
                    combined = np.stack((channel2, channel1, background), axis=-1)
                    pred_img = Image.fromarray(combined)
                    filename = str(i+1) + '_' + str(idx) + '.png'
                    save_path = os.path.join(pred_png_path, filename)
                    pred_img.save(save_path)

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg),
        ('VOE', voe_s.avg),
        ('RVD', RVD_s.avg),
        ('FNR', fnr_s.avg),
        ('FPR',fpr_s.avg),
        ('ASSD',assd_s.avg),
        ('RMSD',rmsd_s.avg),
        ('MSD', msd_s.avg),
        ('Accuracy', acc_s.avg),
        ('F1Score', f1_s.avg),
        ('Precision', Precision_s.avg),
        ('Recall', Recall_s.avg)
    ])

    print(
        'test_loss %.4f - test_iou %.4f - test_dice_1 %.4f - test_dice_2 %.4f'
        % (
            log['loss'], log['iou'], log['dice_1'], log['dice_2']
        )
    )
    print(
        'VOE %.4f - RVD %.4f - FNR %.4f - FPR %.4f - ASSD %.4f - F1 Score %.4f'
        % (
            log['VOE'],log['RVD'], log['FNR'], log['FPR'], log['ASSD'],log['F1Score']
        )
    )
    print(
        'RMSD %.4f - MSD %.4f - Accuracy %.4f - Precision %.4f - Recall %.4f'
        % (
           log['RMSD'],log['MSD'],log['Accuracy'],log['Precision'],log['Recall']
        )
    )
    #save in csv
    df = pd.DataFrame([log.values()], columns=log.keys())
    df.to_csv(os.path.join(csv_path, 'miccai_official_evaluation.csv'), header=True,index=False)



if __name__ == '__main__':
    main()