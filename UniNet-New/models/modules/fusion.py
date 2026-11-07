# 原网络的跨模态融合位置：MFF。
# x = img + depth + img * depth；通道/空间权重经像素门控后加权两路输入。
# img/depth 是原变量名，实际辅助模态由数据目录决定。
import torch.nn as nn
from .attention import ChannelAttention, SpatialAttention, PixelAttention

class MFF(nn.Module):
    def __init__(self, in_channel, dim,reduction_ratio=16,pixel_dim=32):
        super(MFF, self).__init__()
         
        self.ChannelGate = ChannelAttention(in_channel, reduction_ratio)
        self.SpatialGate = SpatialAttention()
        self.PixelGate = PixelAttention(pixel_dim)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
    def forward(self, img, depth):
        x = img + depth + (img * depth)
        cattn=self.ChannelGate(x)
        sattn=self.SpatialGate(x)
        pattn1=cattn+sattn
        pattn2=self.PixelGate(x,pattn1)
        result=x + pattn2 * img+(1 - pattn2) * depth
        result=self.conv(result)
        return result
