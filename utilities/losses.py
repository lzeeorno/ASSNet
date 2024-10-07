import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss_lits2017(nn.Module):
    def __init__(self):
        super(BCEDiceLoss_lits2017, self).__init__()

    def forward(self, input, target):

        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        l2_reg = 0.9989
        input = torch.sigmoid(input)
        num = target.size(0)

        input_1 = input[:,1,:,:]
        input_2 = input[:,2,:,:]
        target_1 = target[:,1,:,:]
        target_2 = target[:,2,:,:]

        input_1 = input_1.view(num, -1)
        input_2 = input_2.view(num, -1)

        target_1 = target_1.view(num, -1)
        target_2 = target_2.view(num, -1)

        intersection_1 = (input_1 * target_1)
        intersection_2 = (input_2 * target_2)

        dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)
        dice_2 = (2. * intersection_2.sum(1) + smooth) / (input_2.sum(1) + target_2.sum(1) + smooth)

        dice_1 = (1 - dice_1.sum() / num)
        dice_2 = (1 - dice_2.sum() / num)

        mean_dice = (dice_1+dice_2)/2.0

        return bce + mean_dice


class newBCEDiceLoss(nn.Module):
    def __init__(self):
        super(newBCEDiceLoss, self).__init__()

    def forward(self, input, target):
        # input = torch.sigmoid(input)
        # 将灰度图自动加入one-hot 编码,但是如果你已经手动加入了就会导致[10, 3, 512, 512]->[10, 3, 512, 512, 3]
        # target_onehot = F.one_hot(target.long(), num_classes=3)  # [6, 3, 512, 512]
        # print(input.shape, target.shape)
        # print(target_onehot.shape)
        # quit()
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        l2_reg = 0.9989
        input = torch.sigmoid(input)
        num = target.size(0)

        input_1 = input[:,1,:,:]
        input_2 = input[:,2,:,:]
        target_1 = target[:,1,:,:]
        target_2 = target[:,2,:,:]

        input_1 = input_1.view(num, -1)
        input_2 = input_2.view(num, -1)

        target_1 = target_1.view(num, -1)
        target_2 = target_2.view(num, -1)

        intersection_1 = (input_1 * target_1)
        intersection_2 = (input_2 * target_2)

        dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)
        dice_2 = (2. * intersection_2.sum(1) + smooth) / (input_2.sum(1) + target_2.sum(1) + smooth)

        dice_1 = (1 - dice_1.sum() / num)
        dice_2 = (1 - dice_2.sum() / num)

        # mean_dice = (dice_1+dice_2)/2.0
        mean_dice = dice_1*l2_reg*math.pi*0.1 + dice_2*(1-l2_reg*math.pi*0.1)
        # bce_dice_loss = bce - torch.log(mean_dice)
        return bce + mean_dice


class BCEDiceLoss_synapse(nn.Module):
    def __init__(self):
        super(BCEDiceLoss_synapse, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        l2_reg = 0.9989
        input = torch.sigmoid(input)
        num = target.size(0)
        # print('loss function:')
        # print(input.shape)
        # print(target.shape)
        # quit()
        # 初始化 Dice 损失
        dice_loss = 0.0
        for i in range(1, 9):  # 从 1 到 8，忽略背景0
            input_i = input[:, i, :, :].view(num, -1)
            target_i = target[:, i, :, :].view(num, -1)

            intersection = (input_i * target_i).sum(1)
            dice_i = (2. * intersection + smooth) / (input_i.sum(1) + target_i.sum(1) + smooth)
            dice_loss += (1 - dice_i).mean()  # 计算每个器官的 Dice 损失并累加

        mean_dice = dice_loss / 8.0  # 计算平均 Dice 损失

        return bce + mean_dice



class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss



class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, weight=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.weight = weight

    def forward(self, labeled_features, unlabeled_features):
        # Flatten the feature maps
        labeled_features = labeled_features.view(labeled_features.size(0), -1)
        unlabeled_features = unlabeled_features.view(unlabeled_features.size(0), -1)

        # Normalize the feature maps
        labeled_features = F.normalize(labeled_features, p=2, dim=1)
        unlabeled_features = F.normalize(unlabeled_features, p=2, dim=1)

        # Calculate the Euclidean distance between labeled and unlabeled features
        euclidean_distance = F.pairwise_distance(labeled_features, unlabeled_features)

        # Calculate the contrastive loss
        contrastive_loss = torch.mean((euclidean_distance ** 2))

        return contrastive_loss * self.weight