import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


def init_weight(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)

class ResBottleNeck(nn.Module):

    def __init__(self, inplanes, planes, expansion=4, downsample=False, stride=1):
        super(ResBottleNeck, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, expansion*planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(expansion*planes),
            nn.Dropout(0.2)
            )

        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expansion*planes))
        else:
            self.shortcut = nn.Sequential()

        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.residual(x) + self.shortcut(x)
        x = self.ReLU(x)

        return x

class SEBottleNeck(ResBottleNeck):

    def __init__(self, inplanes, planes, expansion=4, downsample=False, stride=1, r=4):
        # inherit ResBottleNeck
        ResBottleNeck.__init__(self, inplanes, planes, expansion, downsample, stride)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(expansion*planes, expansion*planes//r),
            nn.ReLU(inplace=True),
            nn.Linear(expansion*planes//r, expansion*planes),
            nn.Sigmoid()
        )

    def forward(self, x):

        x_res = self.residual(x)
        x_se = self.squeeze(x_res).view(x_res.shape[0], -1)
        x_se = self.excitation(x_se).view(x_res.shape[0],-1, 1, 1) # [B, C, 1, 1]

        x = x_res * x_se + self.shortcut(x)
        x = self.ReLU(x)

        return x

class MSEBottleNeck(ResBottleNeck):

    def __init__(self, inplanes, planes, expansion=4, downsample=False, stride=1, r=4):
        # inherit ResBottleNeck
        ResBottleNeck.__init__(self, inplanes, planes, expansion, downsample, stride)

        self.squeeze = nn.AdaptiveMaxPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(expansion*planes, expansion*planes//r),
            nn.ReLU(inplace=True),
            nn.Linear(expansion*planes//r, expansion*planes),
            nn.Sigmoid()
        )

    def forward(self, x):

        x_res = self.residual(x)
        x_se = self.squeeze(x_res).view(x_res.shape[0], -1)
        x_se = self.excitation(x_se).view(x_res.shape[0],-1, 1, 1) # [B, C, 1, 1]

        x = x_res * x_se + self.shortcut(x)
        x = self.ReLU(x)

        return x

class WSEBottleNeck(ResBottleNeck):

    def __init__(self, inplanes, planes, expansion=4, downsample=False, stride=1, r=8):
        # inherit ResBottleNeck
        ResBottleNeck.__init__(self, inplanes, planes, expansion, downsample, stride)
        # self.squeeze = nn.Linear((2048//planes)**2, 1, bias=False)
        self.weighted_gap = nn.Parameter(torch.ones(1, (2048//planes)**2), requires_grad=True)
        self.excitation = nn.Sequential(
            nn.Linear(expansion*planes, expansion*planes//r),
            nn.ReLU(inplace=True),
            nn.Linear(expansion*planes//r, expansion*planes),
            nn.Sigmoid()
        )

    def forward(self, x):

        x_res = self.residual(x) # [B, C, W, H]
        x_se = x_res.view(x_res.shape[0], x_res.shape[1], -1)
        x_se = F.linear(x_se, self.weighted_gap).view(x_res.shape[0], -1)
        x_se = self.excitation(x_se).view(x_res.shape[0], -1, 1, 1) # [B, C, 1, 1]
        x = x_res * x_se + self.shortcut(x)
        x = self.ReLU(x)
        return x

class DSEBottleNeck(ResBottleNeck):

    def __init__(self, inplanes, planes, expansion=4, downsample=False, stride=1, r=4):
        # inherit ResBottleNeck
        ResBottleNeck.__init__(self, inplanes, planes, expansion, downsample, stride)

        self.squeeze1 = nn.AdaptiveAvgPool2d(1)
        self.excitation1 = nn.Sequential(
            nn.Linear(expansion*planes, expansion*planes//r),
            nn.ReLU(inplace=True),
            nn.Linear(expansion*planes//r, expansion*planes),
            nn.Sigmoid()
        )
        self.excitation2 = nn.Sequential(
            nn.Linear((2048//planes)**2, (2048//planes)**2//r),
            nn.ReLU(inplace=True),
            nn.Linear((2048//planes)**2//r, (2048//planes)**2),
            nn.Sigmoid()
        )

    def forward(self, x):

        x_res = self.residual(x)
        x_se1 = self.squeeze1(x_res).view(x_res.shape[0], -1)
        x_se1 = self.excitation1(x_se1).view(x_res.shape[0],-1, 1, 1) # [B, C, 1, 1]

        x_se2 = torch.mean(x_res, dim=1).view(x_res.shape[0], -1) # [B, W, H]
        x_se2 = self.excitation2(x_se2).view(x_res.shape[0], 1, x_res.shape[2], x_res.shape[3])
        x = x_res * x_se1 + x_res * x_se2 + self.shortcut(x)
        x = self.ReLU(x)
        return x

class ResNet(nn.Module):

    def __init__(self, in_ch = 3,num_classes=100, block=ResBottleNeck, inplanes=32, expansion=2):
        super().__init__()

        self.expansion = expansion
        self.inplanes = inplanes

        self.conv_init = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inplanes, kernel_size=3, bias=False, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2),
        )
        #n_layers = [3, 6, 6, 3]
        n_layers = [2, 5, 5, 2]
        planes_list = [x*inplanes for x in [1, 2, 4, 8]]
        stride_list = [1, 2, 2, 2]

        # conv_1
        self.layers = [self.conv_init]
        # conv_2, 3, 4, 5
        for i in range(len(n_layers)):
            self.layers.append(self._make_layer(block, planes_list[i], n_layers[i], stride_list[i]))
        # average pool & fully connected layer
        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.layers = nn.ModuleList(self.layers)
        self.fc = nn.Linear(inplanes*expansion*8, num_classes)        
        #planes_list = [x*inplanes for x in [1, 2, 4, 8]]

        self.dropout = nn.Dropout(0.2)

    def _make_layer(self, block, planes, n_block, stride):

        layers = []
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = True
        layers.append(block(self.inplanes, planes, self.expansion, downsample=downsample, stride=stride))
        self.inplanes = planes * self.expansion
        for i in range(n_block-1):
            layers.append(block(self.inplanes, planes, self.expansion, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        feature = x.view(x.shape[0], -1)
        #feature = self.dropout(feature)
        x = self.fc(feature)

        return F.sigmoid(x), feature

