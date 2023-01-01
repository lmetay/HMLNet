
from copy import copy
import torch
import torch.nn as nn
from .encoder import Backbone
from .Attention import multiscale_att,Channel_att
from .decoder import decoder 
import torch.nn.functional as F

class CDNet(nn.Module):
    def __init__(self, in_c, out_c): 
        super(CDNet,self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.backbone = Backbone(in_c=self.in_c, f_c=self.out_c)

        self.Att_ftr1_sa = multiscale_att(in_channels=self.out_c)
        self.Att_ftr1_sb = multiscale_att(in_channels=self.out_c)
        self.Att_ftr1_ca = Channel_att()
        self.Att_ftr1_cb = Channel_att()

        self.Att_ftr2_sa = multiscale_att(in_channels=self.out_c)
        self.Att_ftr2_sb = multiscale_att(in_channels=self.out_c)
        self.Att_ftr2_ca = Channel_att()
        self.Att_ftr2_cb = Channel_att()

        self.Att_ftr3_sa = multiscale_att(in_channels=self.out_c)
        self.Att_ftr3_sb = multiscale_att(in_channels=self.out_c)
        self.Att_ftr3_ca = Channel_att()
        self.Att_ftr3_cb = Channel_att()

        self.Att_ftr4_sa = multiscale_att(in_channels=self.out_c)
        self.Att_ftr4_sb = multiscale_att(in_channels=self.out_c)
        self.Att_ftr4_ca = Channel_att()
        self.Att_ftr4_cb = Channel_att()

        self.decoder = decoder(self.out_c)

    
    def forward(self, input1, input2):
        f_A = self.backbone(input1)
        f_B = self.backbone(input2)

        ftr_A = self.decoder(f_A[0], f_A[1], f_A[2], f_A[3], f_A[4])
        ftr_B = self.decoder(f_B[0], f_B[1], f_B[2], f_B[3], f_B[4])


        f1_A = self.Att_ftr1_sa(ftr_A[0])
        f1_B = self.Att_ftr1_sb(ftr_B[0])
        f1_A = self.Att_ftr1_ca(f1_A)
        f1_B = self.Att_ftr1_cb(f1_B)

        f2_A = self.Att_ftr2_sa(ftr_A[1])
        f2_B = self.Att_ftr2_sb(ftr_B[1])
        f2_A = self.Att_ftr2_ca(f2_A)
        f2_B = self.Att_ftr2_cb(f2_B)

        f3_A = self.Att_ftr3_sa(ftr_A[2])
        f3_B = self.Att_ftr3_sb(ftr_B[2])
        f3_A = self.Att_ftr3_ca(f3_A)
        f3_B = self.Att_ftr3_cb(f3_B)

        f4_A = self.Att_ftr4_sa(ftr_A[3])
        f4_B = self.Att_ftr4_sb(ftr_B[3])
        f4_A = self.Att_ftr4_ca(f4_A)
        f4_B = self.Att_ftr4_cb(f4_B)


        f_A = [f1_A, f2_A, f3_A, f4_A]
        f_B = [f1_B, f2_B, f3_B, f4_B]


        DI_map = []
        for i in range(0, len(f_A)):
            dist = F.pairwise_distance(f_A[i], f_B[i], keepdim=True)
            DI_map += dist
       
        return DI_map, len(f_A)

if __name__ == '__main__':
    input=torch.randn(8,3,256,256).cuda()
    eca = CDNet(in_c=3, out_c=128).cuda()
    output=eca(input, input)
    print(output.shape)