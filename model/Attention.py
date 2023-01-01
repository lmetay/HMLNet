import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import matplotlib.pyplot as plt 


class multiscale_att(nn.Module):
    def __init__(self, in_channels):
        super(multiscale_att, self).__init__()
        self.in_channels = in_channels
        self.MultiScale = MultiScale(self.in_channels, self.in_channels)
        self.SpatialAttention_ksize = SpatialAttention_ksize()

    def forward(self, ftr):
        """先多尺度再attention"""
        f_3_3, f_5_5, f_7_7 = self.MultiScale(ftr) # ftr.size() = f_3_3.size() = f_5_5.size() = f_7_7.size() = [B, C, H, W]
        ftr_sat1 = self.SpatialAttention_ksize(f_3_3, ksize=3) # [B, 1, H, W]
        ftr_sat2 = self.SpatialAttention_ksize(f_5_5, ksize=3) # [B, 1, H, W]
        ftr_sat3 = self.SpatialAttention_ksize(f_7_7, ksize=3) # [B, 1, H, W]
        sat = (ftr_sat1 + ftr_sat2 + ftr_sat3)/3               # [B, 1, H, W]
        out = ftr *sat + ftr                         
        return out





class Channel_att(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        """
            inputs :
                x : input feature maps   B x C x W x H
            returns :
                out: 
                attention: B X N X N (N is Width*Height)
        """
        out=self.gap(input) # B x C x 1 x 1
        out=out.squeeze(-1).permute(0,2,1) # B x 1 x C 
        out=self.conv(out) # B x 1 x C 
        out=self.sigmoid(out) # B x 1 x C 
        out=out.permute(0,2,1).unsqueeze(-1) # B x C x 1 x 1
        return input*out.expand_as(input)


class MultiScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScale, self).__init__()
        self.conv_3_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.conv_5_5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2, dilation=1),  
            nn.ReLU(inplace=True)
        )
        self.conv_7_7 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=3, dilation=1),  
            nn.ReLU(inplace=True)
        )
    def forward(self, ftr):
        ftr_3_3 = self.conv_3_3(ftr)
        ftr_5_5 = self.conv_5_5(ftr)
        ftr_7_7 = self.conv_7_7(ftr)
        return ftr_3_3, ftr_5_5, ftr_7_7
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SpatialAttention_ksize(nn.Module):
    def __init__(self):
        super(SpatialAttention_ksize, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, dilation = 1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=2, dilation = 2, bias=False)
        self.conv7 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=3, dilation = 3, bias=False)
        
    def forward(self, ftr, ksize):
        # ftr: [B, C, H, W]
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True) # [B, 1, H, W]
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True) # [B, 1, H, W]
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1) # [B, 2, H, W]
        if ksize == 3: am = torch.sigmoid(self.conv3(ftr_cat)) # [B, 1, H, W]
        if ksize == 5: am = torch.sigmoid(self.conv5(ftr_cat)) # [B, 1, H, W]
        if ksize == 7: am = torch.sigmoid(self.conv7(ftr_cat)) # [B, 1, H, W]
        return am

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



























