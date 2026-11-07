# 训练调度：初始化进程组、构造模型/优化器、逐轮训练验证及保存。
import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from configs import defaults as cfg
from datasets import get_loader
from models import UniNet
from utils.paths import split_paths
from utils.optimization import adjust_lr
from utils.checkpoint import load_checkpoint
from .train_one_epoch import train_one_epoch
from .evaluate import validate

def run_training(opt):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', world_size=1, rank=0)
    # 确保保存路径以斜杠结尾
    if not opt.save_path.endswith('/'):
        opt.save_path += '/'
    os.makedirs(opt.save_path, exist_ok=True)

    logging.basicConfig(filename=opt.save_path+'Plog.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("USOD10K-Train")

    model = UniNet().cuda()
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 1
    best_mae = float('inf')

    # 加载检查点（如果提供）
    if opt.resume:
        print(f"Attempting to load checkpoint from {opt.resume}")
        try:
            start_epoch, _ = load_checkpoint(model, optimizer, opt.resume)
            print(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch}")

            # 如果从检查点恢复，尝试加载之前的最佳MAE
            checkpoint_dir = os.path.dirname(opt.resume)
            best_model_path = os.path.join(checkpoint_dir, cfg.BEST_MODEL_NAME)
            if os.path.exists(best_model_path):
                print(f"Found previous best model at {best_model_path}")

        except Exception as e:
            print(f"Failed to load checkpoint: {str(e)}")
            print("Starting training from scratch")
            start_epoch = 1

    train_image_root, train_gt_root, train_pianzhen_root = split_paths(opt.data_root, opt.train_split)
    train_loader = get_loader(train_image_root, train_gt_root, train_pianzhen_root, batch_size=opt.batchsize, image_size=opt.trainsize, num_workers=cfg.NUM_WORKERS)
    total_step = len(train_loader)

    val_image_root, val_gt_root, val_pianzhen_root = split_paths(opt.data_root, opt.val_split)
    val_loader = get_loader(val_image_root, val_gt_root, val_pianzhen_root, batch_size=opt.batchsize, image_size=opt.trainsize, num_workers=cfg.NUM_WORKERS, augment=False)

    writer = SummaryWriter(opt.save_path + "SummaryWriter")

    print('--------------------training----------------------')
    print(f'Starting from epoch: {start_epoch}')

    for epoch in range(start_epoch, opt.epoch+1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)

        train_one_epoch(train_loader, model, optimizer, epoch, opt, total_step, writer)
        current_mae = validate(val_loader, model, epoch, opt, writer)

        if current_mae < best_mae:
            best_mae = current_mae
            torch.save(model.state_dict(), opt.save_path + cfg.BEST_MODEL_NAME)
            print(f"New best model saved with MAE: {best_mae:.4f}")

            # 同时保存完整的检查点
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_mae,
                'best_mae': best_mae
            }
            torch.save(checkpoint, opt.save_path + f'P_best_epoch_{epoch}.pth')

    writer.close()
