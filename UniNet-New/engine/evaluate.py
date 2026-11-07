# 验证集MAE按原有批均值与样本数加权方式计算。
import torch
from datetime import datetime
import logging

def validate(val_loader, model, epoch, opt, writer):
    model.eval()
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for step, data in enumerate(val_loader):
            imgs, gts, pianzhen = data
            imgs = imgs.cuda()
            gts = gts.cuda()
            pianzhen = pianzhen.cuda()
            input_fea = torch.cat((imgs, pianzhen), dim=0)

            _, pre = model(input_fea)
            pre = torch.sigmoid(pre)

            # Calculate MAE (Mean Absolute Error)
            mae = torch.abs(pre - gts).mean()
            total_mae += mae.item() * imgs.size(0)
            total_samples += imgs.size(0)

    avg_mae = total_mae / total_samples
    print(
        '[{}] => [Epoch Num: {:03d}/{:03d}] => [Val MAE: {:0.4f}]'.
        format(datetime.now(), epoch, opt.epoch, avg_mae))
    logging.info(
        '#VAL#:Epoch [{:03d}/{:03d}], MAE: {:0.4f}'.
        format(epoch, opt.epoch, avg_mae))

    writer.add_scalar("Val_MAE", avg_mae, global_step=epoch)
    return avg_mae
