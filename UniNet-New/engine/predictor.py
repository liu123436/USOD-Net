# 显著图推理导出：保持阶段/最终预测相加、sigmoid及逐图min-max后处理。
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models import UniNet
from datasets import test_dataset
from utils.paths import split_paths

def run_inference(opt):
    for _data_name in [opt.test_split]:
        data_path = os.path.join(opt.data_root, _data_name)
        save_path = os.path.join(opt.save_path, _data_name)
        model = UniNet()

        # 加载 checkpoint 文件
        checkpoint = torch.load(opt.pth_path)

        # 判断 checkpoint 是否包含额外的键
        if 'model_state_dict' in checkpoint:
            # 如果是完整的 checkpoint，提取模型权重
            model_state_dict = checkpoint['model_state_dict']
        else:
            # 如果是单独的模型权重，直接使用
            model_state_dict = checkpoint

        # 加载模型权重
        model.load_state_dict(model_state_dict)
        model.cuda()
        model.eval()

        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)
        print("Save path:", save_path)
    
        image_root, gt_root, depth_root = split_paths(opt.data_root, _data_name)

        print('root', image_root, depth_root, gt_root)
        test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
        print('****', test_loader.size)
        for i in range(test_loader.size):
            image, gt, depth, name = test_loader.load_data()
            print('***name', name)
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()

            x_in = torch.cat((image, depth), dim=0)

            P1, P2 = model(x_in)
            res = F.upsample(P1[-1] + P2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype(np.uint8)

            # 提取文件名并确保以 .png 结尾
            name = os.path.basename(name)
            if not name.endswith('.png'):
                name += '.png'

            # 拼接保存路径
            save_image_path = os.path.join(save_path, name)

            # 保存图片
            success = cv2.imwrite(save_image_path, res)
            if not success:
                print(f"Failed to save image: {save_image_path}")
            else:
                print(f"Saved image: {save_image_path}")
