
import torch
import torch.nn as nn
import torch.nn.functional as F


class decoder(nn.Module):
    def __init__(self, fc):
        super(decoder, self).__init__()
        self.fc = fc
        self.dr1 = DR(128, 96) #64, 96
        self.dr2 = DR(256, 96) #64, 96
        self.dr3 = DR(512, 96) #128, 96
        self.dr4 = DR(1024, 96) #256 96
        self.dr5 = DR(2048, 96) #512, 96
        #nn.Conv2d的输入通道为 96*4 即384
        
        self.conv = nn.Sequential(nn.Conv2d(480, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(self.fc),
                                       nn.ReLU(),
                                       ) 
        self.pyramid1 = nn.Sequential(nn.Conv2d(in_channels=self.fc, out_channels=self.fc, kernel_size=3, stride=2, padding=1, dilation=1), 
                                      nn.BatchNorm2d(self.fc),  
                       	              nn.ReLU(inplace=True))
        self.pyramid2 = nn.Sequential(nn.Conv2d(in_channels=self.fc, out_channels=self.fc, kernel_size=3, stride=2, padding=1, dilation=1), 
                                      nn.BatchNorm2d(self.fc),  
                       	              nn.ReLU(inplace=True))
        self.pyramid3 = nn.Sequential(nn.Conv2d(in_channels=self.fc, out_channels=self.fc, kernel_size=3, stride=2, padding=1, dilation=1), 
                                      nn.BatchNorm2d(self.fc),  
                       	              nn.ReLU(inplace=True))

        self._init_weight()


    def forward(self, f1, f2, f3, f4, f5):
        f1 = self.dr1(f1)
        f2 = self.dr2(f2)
        f3 = self.dr3(f3)
        f4 = self.dr4(f4)
        f5 = self.dr5(f5)

        #upsample
        f2 = F.interpolate(f2, size=f1.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, size=f1.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(f4, size=f1.size()[2:], mode='bilinear', align_corners=True)
        f5 = F.interpolate(f5, size=f1.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((f1, f2, f3, f4, f5), dim=1)
        x = self.conv(x) #[B, 128, 128, 128]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x1 = self.pyramid1(x)  #[128, 128, 128]
        x2 = self.pyramid2(x1)  #[128, 64, 64]
        x3 = self.pyramid3(x2)  #[128, 32, 32]

        return x,x1,x2,x3

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)# kaiming 初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# DR 和 Decoder 类完成的是ResNet的特征（low_level_feat*) --> Relu(BN(Conv2d(1x1))) --> upsample
#                                    --> concatenation --> Conv2d(3x3 and 1x1)
class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x



if __name__ == '__main__':
    f1=torch.randn(8,128,128,128).cuda()
    f2=torch.randn(8,256,128,128).cuda()
    f3=torch.randn(8,512,64,64).cuda()
    f4=torch.randn(8,1024,32,32).cuda()
    f5=torch.randn(8,2048,16,16).cuda()
    eca =  decoder(128).cuda()
    output=eca(f1,f2,f3,f4,f5)
    print(output.shape)




