# 主网络：共享编码、多尺度边缘增强、跨模态融合、迭代解码与浅层输出分支。
# 模块成员名与运算顺序保持不变；输入/编码器配置由 configs 统一管理。
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.defaults import BACKBONE, PRETRAINED_PATH
from .backbones import pvt
from .modules.common import BasicConv2d
from .modules.aspp import ASPP
from .modules.fusion import MFF
from .modules.refinement import Enhanced_GGA, RFD
from .modules.edges import MultiScaleEdgeFusion

class UniNet(nn.Module):
    def __init__(self, channel=32, kernel_size=3, reduction=4, bias=False, act=nn.PReLU(), n_resblocks=2, iteration=3):
        super(UniNet, self).__init__()

        self.backbone = getattr(pvt, BACKBONE)()  # [64, 128, 320, 512]
        path = str(PRETRAINED_PATH)
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.iteration = iteration

        # 四级上下文提取/通道投影，随后由 edge_fusion 增强边缘。
        self.DEConv_4 = ASPP(64)
        self.DEConv_3 = ASPP(128)
        self.DEConv_2 = ASPP(320)
        self.DEConv_1 = ASPP(512)

        self.edge_fusion = MultiScaleEdgeFusion(channels=[channel, channel, channel, channel])

        # 四级跨模态融合，成员名也对应检查点中的参数键名。
        self.mff_4 = MFF(channel,channel)
        self.mff_3 = MFF(channel,channel)
        self.mff_2 = MFF(channel,channel)
        self.mff_1 = MFF(channel,channel)

        # 迭代解码中的引导门控；首轮跳过门控，后续轮次启用。
        self.gate_1 = Enhanced_GGA(channel, channel)
        self.gate_2 = Enhanced_GGA(channel, channel)
        self.gate_3 = Enhanced_GGA(channel, channel)

        self.rfd_1 = RFD(channel, kernel_size, reduction, bias, act, n_resblocks)
        self.rfd_2 = RFD(2 * channel, kernel_size, reduction, bias, act, n_resblocks)
        self.rfd_3 = RFD(3 * channel, kernel_size, reduction, bias, act, n_resblocks)

        self.gate_conv = nn.Sequential(
            BasicConv2d(32, 1, 1),
            nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        )
        self.gate_conv_1 = BasicConv2d(32, 1, 1)
        self.gate_conv_2 = BasicConv2d(64, 1, 1)

        self.unsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.out = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.pred = nn.Conv2d(channel, 1, 1)

        self.Fus = ASPP(channel)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_pred = nn.Conv2d(channel, 1, 1)
        self.final = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):  # 孪生网络，这里是batch拼接后的img和depth

        # 输入 [2B,3,H,W]：前 B 张 RGB、后 B 张辅助图，共享同一个编码器。
        # H=W=352 时，下面四级空间尺寸依次为 88、44、22、11。
        pvt = self.backbone(x)
        x4 = pvt[0]
        x3 = pvt[1]
        x2 = pvt[2]
        x1 = pvt[3]

        x4 = self.DEConv_4(x4)
        x3 = self.DEConv_3(x3)
        x2 = self.DEConv_2(x2)
        x1 = self.DEConv_1(x1)

        x1, x2, x3, x4 = self.edge_fusion(x1, x2, x3, x4)

        x4_img, x4_depth = torch.chunk(x4, 2, dim=0)
        x3_img, x3_depth = torch.chunk(x3, 2, dim=0)
        x2_img, x2_depth = torch.chunk(x2, 2, dim=0)
        x1_img, x1_depth = torch.chunk(x1, 2, dim=0)

        # 深/中尺度迭代解码分支；每轮预测按原顺序加入 stage_pred。
        stage_pred = list()
        coarse_pred = None  # 阶段预测结果
        for iter in range(self.iteration):
            x1 = self.mff_1(x1_img, x1_depth)
            if coarse_pred == None:
                x1 = x1
            else:
                coarse_pred = self.gate_conv(coarse_pred)
                x1 = self.gate_1(x1, coarse_pred)
            x2_feed = self.rfd_1(x1)
            x2 = self.mff_2(x2_img, x2_depth)
            if iter > 0:
                x2_gate = self.unsample_2(self.gate_conv_1(x2_feed))
                x2 = self.gate_2(x2, x2_gate)
            x3_feed = self.rfd_2(torch.cat((x2, self.unsample_2(x2_feed)), dim=1))
            x3 = self.mff_3(x3_img, x3_depth)
            if iter > 0:
                x3_gate = self.unsample_2(self.gate_conv_2(x3_feed))
                x3 = self.gate_3(x3, x3_gate)
            x4_feed = self.rfd_3(torch.cat((x3, self.unsample_2(x3_feed)), dim=1))
            coarse_pred = self.out(x4_feed)
            out_map = self.pred(coarse_pred)  # 352输入时为 [B, 1, 44, 44]
            pred = F.interpolate(out_map, scale_factor=8, mode='bilinear')
            stage_pred.append(pred)

        # 最终输出分支独立使用浅层融合特征，不接入迭代分支的 coarse_pred。
        # 数据流：mff_4 → downsample → Fus → out_pred → final。
        x4 = self.mff_4(x4_img, x4_depth)
        x4_out = self.downsample(x4)
        refined_pred = self.Fus(x4_out)
        pred2 = self.out_pred(refined_pred)
        final_pred=self.final(pred2)
        return stage_pred, final_pred
