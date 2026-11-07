# USOD10K目录接线；读取/增强/归一化逻辑由datasets层负责。
import os

def split_paths(data_root, split):
    root = os.path.join(data_root, split)
    # 原数据集类使用字符串拼接，路径必须保留末尾分隔符。
    rgb = os.path.join(root, split + '-RGB', '')
    gt = os.path.join(root, split + '-GT', '')
    depth = os.path.join(root, split + '-depth', '')
    return rgb, gt, depth
