import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import matplotlib.pyplot as plt 

class Backbone(nn.Module):
    def __init__(self, backbone='res2net50mod',in_c=3, f_c=64):
        self.in_c = in_c
        super(Backbone, self).__init__()
        self.module = mynet3(backbone=backbone, f_c=f_c, in_c=self.in_c)
    def forward(self, input):
        return self.module(input)

class mynet3(nn.Module):
    def __init__(self, backbone='res2net50mod', f_c=64, freeze_bn=False, in_c=3):
        super(mynet3, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.in_c = in_c
        self.endoder1 = Res2Net1(Bottle2neck, self.in_c, BatchNorm)
        self.endoder2 = Res2Net2(Bottleneck, BatchNorm)
        self.decoder = Decoder(f_c,BatchNorm)

        if freeze_bn:
            self.freeze_bn()
    
    def forward(self, input):
        f1, f2 = self.endoder1(input)
        f3, f4, f5 = self.endoder2(f2)
        return [f1, f2, f3, f4, f5]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval() 





class Res2Net1(nn.Module):

    def __init__(self, block1, in_c, BatchNorm):
        self.inplanes = 64
        self.in_c = in_c
        super(Res2Net1, self).__init__()

        self.conv1 = nn.Conv2d(self.in_c, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer = self._make_layer(block1, 32, 3, stride=1, dilation=1, BatchNorm=BatchNorm)
        self.layer1 = self._make_layer(block1, 64, 2, stride=1, dilation=1, BatchNorm=BatchNorm)
        self._init_weight()
   
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):  
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            ) 
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm, stype='stage'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
        
        return nn.Sequential(*layers)
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        f1 = self.layer(x)
        x = self.maxpool(f1)

        f2 = self.layer1(x) 
 
        return f1,f2

    def _init_weight(self):        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained_model(self, model_path):
        pretrain_dict = model_zoo.load_url(model_path)
        model_dict = {}
        state_dict = self.state_dict() 
        for k, v in pretrain_dict.items():
            if 'layer1.0' in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict) 


class Res2Net2(nn.Module):

    def __init__(self, block2, BatchNorm):
        self.inplanes = 256
        super(Res2Net2, self).__init__()

        self.layer2 = self._make_layer1(block2, 128, 2, stride=2, dilation=1, BatchNorm=BatchNorm)
        self.layer3 = self._make_layer1(block2, 256, 2, stride=2, dilation=1, BatchNorm=BatchNorm)
        self.layer4 = self._make_layer1(block2, 512, 2, stride=2, dilation=1, BatchNorm=BatchNorm)
        self._init_weight()
   
    
    def _make_layer1(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            ) 

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    
    def forward(self, input):
        f3 = self.layer2(input)  
        f4 = self.layer3(f3)  
        f5 = self.layer4(f4)  

        return f3,f4,f5

    def _init_weight(self):        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained_model(self, model_path):
        pretrain_dict = model_zoo.load_url(model_path)
        model_dict = {}
        state_dict = self.state_dict() 
        for k, v in pretrain_dict.items():
            if 'layer2.0' in state_dict:
                model_dict[k] = v
            if 'layer3.0' in state_dict:
                model_dict[k] = v
            if 'layer4.0' in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict) 


class Bottle2neck(nn.Module):
    expansion = 4 

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, 
                BatchNorm=None, baseWidth=26, scale = 4, stype='normal'):

        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(width*scale)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1

        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.stype = stype 
        self.scale = scale
        self.width  = width
    
    def forward(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i==0 or self.stype =='stage':
                sp = spx[i]
            else: 
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i==0:
                out = sp
            else:
                out = torch.cat((out,sp),1)
        if self.scale != 1 and self.stype =='normal': 
            out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype =='stage':
            out = torch.cat((out, self.pool(spx[self.nums])),1)


        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out






class Decoder(nn.Module):
    def __init__(self, fc, BatchNorm):
        super(Decoder, self).__init__()
        self.fc = fc
        self.dr1 = DR(128, 96)
        self.dr2 = DR(256, 96)
        self.dr3 = DR(512, 96)
        self.dr4 = DR(1024, 96)
        self.dr5 = DR(2048, 96)
        self.last_conv = nn.Sequential(nn.Conv2d(480, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(self.fc),
                                       nn.ReLU(),
                                    )
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
        x = self.last_conv(x)
        # x = F.interpolate(x, size=[256,256], mode='bilinear', align_corners=True)
       
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


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

