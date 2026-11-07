# 所有可编辑的实验默认值集中在本文件。
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKBONE = "pvt_v2_b2"
INPUT_SIZE = 352
BATCH_SIZE = 8
PRETRAINED_PATH = PROJECT_ROOT / "weights" / "pvt_v2_b2.pth"

# 服务器上已确认的数据目录；迁移机器时修改此项或使用 --data_root。
DATA_ROOT = "/root/autodl-tmp/data/USOD10K"
TRAIN_SPLIT = "USOD10K-TR"
VAL_SPLIT = "USOD10K-Val"
TEST_SPLIT = "USOD10K-TE"
SAVE_PATH = str(PROJECT_ROOT / "outputs" / "checkpoints")
PREDICTION_PATH = str(PROJECT_ROOT / "outputs" / "predictions")
BEST_MODEL_NAME = "D_UniNet.pth"
CHECKPOINT_PATH = str(PROJECT_ROOT / "outputs" / "checkpoints" / BEST_MODEL_NAME)

# 以下训练默认值沿用源代码；本次没有调整训练策略。
EPOCHS = 100
LEARNING_RATE = 1e-4
OPTIMIZER = "AdamW"
AUGMENTATION = False
GRAD_CLIP = 0.5
DECAY_RATE = 0.1
DECAY_EPOCH = 50
EPOCH_SAVE = 20
NUM_WORKERS = 2
CUDA_VISIBLE_DEVICES = "0"
MASTER_ADDR = "localhost"
MASTER_PORT = "12355"
