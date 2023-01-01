

import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 


class BDCL(nn.Module):
    def __init__(self, alpha):
        super(BDCL, self).__init__()
        self.alpha = alpha
        self.BCL = BCL(margin=2.0)
        self.Dice = dice_loss_binary_class()
    
    def forward(self, distance, label):
        loss_BCL = self.BCL(distance, label)
        loss_Dice = self.Dice(distance, label)
        loss = (1-self.alpha)*loss_BCL + self.alpha*loss_Dice
        return loss



class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """
    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # label = torch.argmax(label, 1).unsqueeze(1).float()
        label[label == 1] = -1
        label[label == 0] = 1

        mask = (label != 255).float()
        distance = distance * mask

        pos_num = torch.sum((label==1).float())+0.0001
        neg_num = torch.sum((label==-1).float())+0.0001

        loss_1 = torch.sum((1+label) / 2 * torch.pow(distance, 2)) /pos_num
        loss_2 = torch.sum((1-label) / 2 *
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        ) / neg_num
        loss = loss_1 + loss_2
        return loss



class dice_bce_loss_binary_class(nn.Module):
    """    Binary    """
    def __init__(self):
        super(dice_bce_loss_binary_class, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.binary_loss = dice_loss_binary_class()
    
    def __call__(self, scores, labels, do_sigmoid=True):
        if len(scores.shape)>3:
            scores = scores.squeeze(1)
        if len(labels.shape)>3:
            labels = labels.squeeze(1)
        if do_sigmoid:
            scores = torch.sigmoid(scores.clone())
        diceloss = self.binary_loss(scores, labels)
        bceloss = self.bce_loss(scores, labels)
        return diceloss#+bceloss #diceloss+


class dice_loss_binary_class(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss_binary_class, self).__init__()
        self.batch = batch
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.00001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred) #交集
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_true, y_pred):
        return self.soft_dice_loss(y_true, y_pred.to(dtype=torch.float32))






class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):  #在消融实验中由mean改为sum
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape == target.shape, "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)
        # predict = torch.sigmoid(predict)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]





