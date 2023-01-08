import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


from utils import Debug



class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=pretrained)
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pretrained = pretrained

        if self.pretrained:
            self.model.eval()
        else:
            self.model.train()
        
        self.level1 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
        )
        self.level2 = self.model.layer1
        self.level3 = self.model.layer2
        self.level4 = self.model.layer3
        self.level5 = self.model.layer4
        
    
    def forward(self, x):
        out1 = self.level1(x)
        out2 = self.level2(out1)
        out3 = self.level3(out2)
        out4 = self.level4(out3)
        out5 = self.level5(out4)
        return (out1, out2, out3, out4, out5)


# Shortcut Convulation module
class SCM(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(SCM, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x):
        feature1 = self.block1(x)
        x = F.relu(torch.sum([feature1, x], dim=1))
        feature2 = self.block2(x)
        return F.relu(torch.sum([feature2, x], dim=1))


# Squeeze Excitation Module
class SEM(nn.Module):
    def __init__(self, in_channels, out_conv2d_channels, out_linear_channels):
        super(SEM, self).__init__()
        self.sem = nn.Sequential(
            nn.Conv2d(in_channels, out_conv2d_channels, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(1, out_linear_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_linear_channels, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.sem(x)
        return x * features

class GateFunction(nn.Module):
    def __init__(self, num_classes):
        super(GateFunction, self).__init__()
        self.num_classes = num_classes

    def forward(self, pixel):        
        return pixel - (1 / self.num_classes) + torch.sum(-pixel * torch.log(pixel))


class PGM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PGM, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), # out_channels = num_classes (C)
            nn.Softmax(dim=1),
            GateFunction(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_lower, f_upper):
        f_upper = F.interpolate(f_upper, scale_factor=2, mode="nearest")
        Debug.print("After up f_upper", f_upper) # 1, 512, 14, 14
        upper_feature = self.block(f_upper)
        Debug.print("After block, upper_feature", upper_feature)
        Debug.print("f_lower shape", f_lower)
        # f_lower => 1, 2048, 7, 7
        # upper_feature 1, 512, 14, 14
        return f_upper + (f_lower * upper_feature)


class DC_PGM(nn.Module):
    def __init__(self, in_channels, out_channels, pgm_out_channels=None):
        super(DC_PGM, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=3)
        self.pgm1 = PGM(in_channels=512, out_channels=pgm_out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=2560, out_channels=out_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        # self.pgm2 = PGM()

        self.conv3 = nn.Conv2d(in_channels=5120, out_channels=out_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        # self.pgm3 = PGM()
        

    def forward(self, x):
        feature1 = self.conv1(x)
        Debug.print("feature1 ", feature1) # 1, 512, 7, 7 after padding=1
        Debug.print("x shape", x) # 1, 2048, 7, 7
        Debug.end()
        last_feature1 = self.pgm1(feature1, x)

        # Debug.print("last_feature1", last_feature1)

        concat_x1 = torch.cat([x, feature1], dim=1)
        feature2 = self.conv2(concat_x1)
        # last_features2 = self.pgm2(feature2, x)

        concat_x2 = torch.cat([x, concat_x1], dim=1)
        concat_x3 = torch.cat([concat_x2, feature2], dim=1)
        feature3 = self.conv3(concat_x3)
        # last_feature3 = self.pgm3(feature3, x)
        return x
        # return torch.cat([last_feature1, last_features2, last_feature3], dim=1)


class L_GCNN(nn.Module):
    def __init__(self, num_classes):
        super(L_GCNN, self).__init__()
        self.resnet = ResNet(pretrained=True)
        self.level1 = nn.Sequential(
            SCM(in_channels=64, out_channels=64),
            SEM(in_channels=64, out_conv2d_channels=64, out_linear_channels=4),
            PGM(in_channels=64, out_channels=num_classes),
            SCM(in_channels=64, out_channels=64),
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
        )
        # self.level2 = nn.Sequential(
        #     SCM(),
        #     SEM(),
        #     PGM(),
        #     SCM(),
        # )
        # self.level3 = nn.Sequential(
        #     SCM(),
        #     SEM(),
        #     PGM(),
        #     SCM(),
        # )
        # self.level4 = nn.Sequential(
        #     SCM(),
        #     SEM(),
        #     PGM(),
        #     SCM(),
        # )
        self.level5 = nn.Sequential(
            DC_PGM(in_channels=2048, out_channels=512, pgm_out_channels=num_classes),
            # SCM(),
        )
    
    def forward(self, x):
        level1_f, level2_f, level3_f, level4_f, level5_f = self.resnet(x)
        return self.level5(level5_f)

if __name__ == "__main__":
    b_input = torch.randn(1, 3, 224, 224)
    model = L_GCNN(3)
    out = model(b_input)
    # print(out.shape)