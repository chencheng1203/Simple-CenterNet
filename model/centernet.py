import math
import torch
import torch.nn as nn
from .resnet import *


class ResnetDecoder(nn.Module):
    '''
    三层反卷积进行上采样，将 feature map 的尺寸扩大 8 倍
    '''
    def __init__(self, in_channels):
        """
        the input image is downsampled with resnet by 32 stride
        """
        super(ResnetDecoder, self).__init__()
        self.in_channels = in_channels
        if in_channels == 512:
            self.planes = [256, 128, 64]
        else:
            self.planes = [1024, 512, 64]
        
        self.kernel_size = [4, 4, 4]

        self.deconv_layers = []
        for i in range(3):
            plane = self.planes[i]
            kernel_size = self.kernel_size[i]
            curr_layer = nn.ConvTranspose2d(self.in_channels, plane, kernel_size, 2, 1)

            self.deconv_layers.append(curr_layer)
            self.deconv_layers.append(nn.BatchNorm2d(plane))
            self.deconv_layers.append(nn.ReLU(inplace=True))
            
            self.in_channels = plane
        
        self.decoder = nn.Sequential(*self.deconv_layers)
        self.weight_init()

    def forward(self, x):
        return self.decoder(x)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DetHead(nn.Module):
    def __init__(self, planes, num_classes):
        super(DetHead, self).__init__()
        self.heat_map = nn.Sequential(nn.Conv2d(64, planes, 3, 1, 1, bias=False),\
                                        nn.BatchNorm2d(planes), 
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(planes, num_classes, 1, 1),
                                      )
        self.wh = nn.Sequential(nn.Conv2d(64, planes, 3, 1, 1, bias=False),\
                                nn.BatchNorm2d(planes), 
                                nn.ReLU(inplace=True),
                                nn.Conv2d(planes, 2, 1, 1)
                                )
        self.offset = nn.Sequential(nn.Conv2d(64, planes, 3, 1, 1, bias=False),\
                                    nn.BatchNorm2d(planes), 
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(planes, 2, 1, 1),
                                    )

        self.weight_init()

    def forward(self, x):
        heatmap = self.heat_map(x).sigmoid()
        wh = self.wh(x)
        offset = self.offset(x)
        return (heatmap, wh, offset)
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CenterNet(nn.Module):
    def __init__(self, depth=18, num_classes=80):
        super(CenterNet, self).__init__()
        self.depth = depth
        self.num_classes = num_classes

        if depth == 18:
            self.backbone = resnet18(pretrained=True, if_include_top=False)
        if depth == 34:
            self.backbone = resnet34(pretrained=True, if_include_top=False)
        if depth == 50:
            self.backbone = resnet50(pretrained=True, if_include_top=False)
        if depth == 101:
            self.backbone = resnet101(pretrained=True, if_include_top=False)
        if depth == 152:
            self.backbone = resnet152(pretrained=True, if_include_top=False)
    
        if depth in [18, 34]:
            in_channels = 512
        else:
            in_channels = 2048

        self.decoder = ResnetDecoder(in_channels)
        self.head = DetHead(64, num_classes=self.num_classes)

    def forward(self, x):
        features = self.decoder(self.backbone(x))
        return self.head(features)

if __name__ == '__main__':
    x = torch.rand(8, 3, 512, 512)
    model = CenterNet(50, num_classes=80)
    outs = model(x)
    for feat in outs:
        print(feat.shape)
        

