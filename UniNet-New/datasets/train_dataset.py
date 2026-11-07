# 配对训练数据：RGB、显著性GT、辅助图（本项目接入depth）。
# 验证可关闭随机增强，继续使用相同的固定预处理。
import os
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from .augmentations import cv_random_flip, randomCrop, randomRotation, colorEnhance, randomPeper_eg

class CamImgTrain(data.Dataset):
    """训练集"""

    def __init__(self, image_root, gt_root, pianzhen_root, image_size, *, augment=True):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.pianzhen = [pianzhen_root + f for f in os.listdir(pianzhen_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.pianzhen = sorted(self.pianzhen)
        self.filter_files()
        self.dataset_size = len(self.images)
        self.image_size = image_size
        self.augment = augment
        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        pianzhen = self.rgb_loader(self.pianzhen[index])

        if self.augment:
            image, gt, pianzhen = cv_random_flip(image, gt,pianzhen)
            image, gt, pianzhen = randomCrop(image, gt, pianzhen)
            image, gt, pianzhen = randomRotation(image, gt, pianzhen)

            image = colorEnhance(image)
            gt, pianzhen = randomPeper_eg(gt, pianzhen)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        pianzhen = self.depth_transform(pianzhen)
        return image, gt, pianzhen

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.dataset_size
