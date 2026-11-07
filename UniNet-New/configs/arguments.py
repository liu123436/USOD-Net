# 参数解析不加载模型、不初始化GPU，支持直接查看 --help。
import argparse
from . import defaults as cfg

def parse_train_args(argv=None):
    parser = argparse.ArgumentParser(description="Train UniNet on USOD10K")
    parser.add_argument('--epoch', type=int, default=cfg.EPOCHS, help='epoch number')
    parser.add_argument('--lr', type=float, default=cfg.LEARNING_RATE, help='learning rate')
    parser.add_argument('--optimizer', type=str, default=cfg.OPTIMIZER, help='choosing optimizer Adam')
    parser.add_argument('--augmentation', default=cfg.AUGMENTATION, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int, default=cfg.BATCH_SIZE, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=cfg.INPUT_SIZE, help='training image size')
    parser.add_argument('--clip', type=float, default=cfg.GRAD_CLIP, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=cfg.DECAY_RATE, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=cfg.DECAY_EPOCH, help='every n epochs decay learning rate')
    parser.add_argument('--data_root', type=str, default=cfg.DATA_ROOT, help='USOD10K dataset root')
    parser.add_argument('--train_split', type=str, default=cfg.TRAIN_SPLIT, help='training split directory')
    parser.add_argument('--val_split', type=str, default=cfg.VAL_SPLIT, help='validation split directory')
    parser.add_argument('--save_path', type=str, default=cfg.SAVE_PATH, help='checkpoint and log directory')
    parser.add_argument('--epoch_save', type=int, default=cfg.EPOCH_SAVE, help='every n epochs to save model')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training')
    return parser.parse_args(argv)

def parse_test_args(argv=None):
    parser = argparse.ArgumentParser(description="Export UniNet saliency predictions on USOD10K")
    parser.add_argument('--testsize', type=int, default=cfg.INPUT_SIZE, help='testing image size')
    parser.add_argument('--pth_path', type=str, default=cfg.CHECKPOINT_PATH, help='trained model checkpoint')
    parser.add_argument('--data_root', type=str, default=cfg.DATA_ROOT, help='USOD10K dataset root')
    parser.add_argument('--test_split', type=str, default=cfg.TEST_SPLIT, help='test split directory')
    parser.add_argument('--save_path', type=str, default=cfg.PREDICTION_PATH, help='prediction output directory')
    return parser.parse_args(argv)
