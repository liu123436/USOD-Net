# 原网络的门控细化与残差解码：Enhanced_GGA、CALayer、RCAB、RFD。
# 门控使用特征与单通道引导图；alpha/beta 是可学习标量。
# 引导图来自解码特征投影，不应直接当作已经归一化的预测概率。
import torch
import torch.nn as nn
from .common import conv

class Enhanced_GGA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super(Enhanced_GGA, self).__init__()
        self.in_channels = in_channels
        
        # 1. 门控注意力生成（利用单通道引导图）
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=1),  # 融合特征和单通道引导图
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),  # 输出注意力图
            nn.Sigmoid()
        )
        
        # 2. 多分支上下文探索（模拟 Focus 模块的 fp/fn）
        # 前景分支（fp）：强调目标区域
        self.fp_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # 背景分支（fn）：抑制背景噪声
        self.fn_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=4, dilation=4),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels, 3, padding=5, dilation=5),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # 3. 可学习标量（alpha/beta）
        self.alpha = nn.Parameter(torch.ones(1))  # 控制前景分支贡献
        self.beta = nn.Parameter(torch.ones(1))   # 控制背景分支贡献
        
        # 4. 输出卷积
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)

    def forward(self, in_feat, low_level_map):
        # Step 1: 生成注意力图（利用单通道引导图）
        attention = self.gate_conv(torch.cat([in_feat, low_level_map], dim=1))
        # Step 2: 特征分割（前景和背景）
        f_feature = in_feat * attention   # 前景特征
        b_feature = in_feat * (1 - attention)  # 背景特征
        
        # Step 3: 多分支处理（模拟 Focus 的 fp/fn）
        fp_out = self.fp_branch(f_feature)
        fn_out = self.fn_branch(b_feature)
        
        # Step 4: 动态融合（类似 Focus 的 refine1/refine2）
        refined_feature = in_feat + self.alpha * fp_out - self.beta * fn_out
        
        # Step 5: 输出
        out_feat = self.out_conv(refined_feature)
        return out_feat


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):  # act = ReLU or PReLU
        super(RCAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


# Residual Feature Decoder
class RFD(nn.Module):
    def __init__(self, channel, kernel_size, reduction, bias, act, n_resblocks):
        super(RFD, self).__init__()
        modules_body = [RCAB(channel, kernel_size, reduction, bias=bias, act=act) for _ in range(n_resblocks)]
        modules_body.append(conv(channel, channel, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
