# 配对数据的DataLoader，参数与数据处理方式保持不变。
import torch.utils.data as data
from .train_dataset import CamImgTrain

def get_loader(image_root, gt_root, eg_root, batch_size, image_size, shuffle=True, num_workers=0, pin_memory=True, *, augment=True):
    # `num_workers=0` for more stable training

    dataset = CamImgTrain(image_root, gt_root, eg_root, image_size, augment=augment)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader
