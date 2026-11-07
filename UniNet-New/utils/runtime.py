# 单卡运行环境；保留源训练脚本的CUDA与NCCL环境值。
import os
from configs import defaults as cfg

def configure_cuda_environment():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    os.environ['MASTER_ADDR'] = cfg.MASTER_ADDR
    os.environ['MASTER_PORT'] = cfg.MASTER_PORT
