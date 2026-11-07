# 单轮训练：两路输入拼接、阶段损失、反向传播、裁剪与参数更新。
# 优化器显式取形参optim；它就是原入口传入的同一个optimizer对象。
import torch
from datetime import datetime
import logging
from utils.losses import structure_loss
from utils.optimization import clip_gradient

def train_one_epoch(train_loader, model, optim, epoch, opt, total_step, writer):
    total_loss = 0
    model.train()
    for step, data in enumerate(train_loader):
        imgs, gts, pianzhen = data
        imgs = imgs.cuda()
        gts = gts.cuda()
        pianzhen = pianzhen.cuda()
        
        input_fea = torch.cat((imgs, pianzhen), dim=0)

        optim.zero_grad()
        stage_pre, pre = model(input_fea)
        stage_loss_list = [structure_loss(out, gts) for out in stage_pre]
        stage_loss = 0
        gamma = 0.2
        for iteration in range(len(stage_pre)):
            stage_loss += (gamma * iteration) * stage_loss_list[iteration]

        map_loss = structure_loss(pre, gts)
        loss = stage_loss + map_loss
        loss.backward()
        clip_gradient(optim, opt.clip)
        optim.step()

        total_loss += loss

        if step % 1000 == 0 or step == total_step:
            print(
                '[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss: {:0.4f}]'.
                format(datetime.now(), epoch, opt.epoch, step, total_step, loss.item()))
            logging.info(
                '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:0.4f}'.
                format(epoch, opt.epoch, step, total_step, loss.item()))

    writer.add_scalar("Train_Loss", total_loss, global_step=epoch)

    save_path = opt.save_path
    if epoch % opt.epoch_save == 0:
        checkpoint = {
            'epoch': epoch + 1,  # 保存下一个epoch作为起始点
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
        }
        # 此分支不写盘；最优模型及完整检查点由 trainer.py 的验证结果分支保存。
        print(f"Epoch {epoch}: periodic checkpoint writing is disabled; best checkpoints are saved after validation.")
