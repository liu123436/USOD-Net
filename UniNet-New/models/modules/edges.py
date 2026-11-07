# 原网络的边缘增强位置：通道均值 → 冻结的四邻域 Laplacian → 本级拼接卷积。
# 随后将较深层结果逐级上采样，相加到较浅层特征。
import torch
import torch.nn as nn

class LaplacianEdge(nn.Module):
    def __init__(self):
        super(LaplacianEdge, self).__init__()
        # Laplacian 算子
        self.laplacian = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 初始化 Laplacian 算子的权重
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.laplacian.weight = nn.Parameter(laplacian_kernel, requires_grad=False)

    def forward(self, x):
        # 计算 Laplacian 边缘
        edge = self.laplacian(x)
        return edge

class MultiScaleEdgeFusion(nn.Module):
    def __init__(self, channels):
        super(MultiScaleEdgeFusion, self).__init__()
        # 边缘提取模块
        self.edge_extract = LaplacianEdge()
        
        # 多尺度特征融合模块
        self.conv1 = nn.Conv2d(channels[0] + 1, channels[0], kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels[1] + 1, channels[1], kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(channels[2] + 1, channels[2], kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(channels[3] + 1, channels[3], kernel_size=3, padding=1, bias=False)
        
        # 上采样和下采样
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

    def forward(self, x1, x2, x3, x4):
        # 提取边缘信息
        edge1 = self.edge_extract(x1.mean(dim=1, keepdim=True))
        edge2 = self.edge_extract(x2.mean(dim=1, keepdim=True))
        edge3 = self.edge_extract(x3.mean(dim=1, keepdim=True))
        edge4 = self.edge_extract(x4.mean(dim=1, keepdim=True))
        
        
        # 多尺度边缘信息融合
        x1 = torch.cat([x1, edge1], dim=1)
        x1 = self.conv1(x1)
        
        x2 = torch.cat([x2, edge2], dim=1)
        x2 = self.conv2(x2)
        
        x3 = torch.cat([x3, edge3], dim=1)
        x3 = self.conv3(x3)
        
        x4 = torch.cat([x4, edge4], dim=1)
        x4 = self.conv4(x4)
        
        # 将边缘信息融合到其他尺度的特征图中
        x2 = x2 + self.upsample(x1)
        x3 = x3 + self.upsample(x2)
        x4 = x4 + self.upsample(x3)
        
        return x1, x2, x3, x4
