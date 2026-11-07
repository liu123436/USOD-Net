# 训练入口；--help不加载模型或初始化GPU。
from configs.arguments import parse_train_args
from utils.runtime import configure_cuda_environment

def main():
    opt = parse_train_args()
    configure_cuda_environment()
    from engine.trainer import run_training
    run_training(opt)

if __name__ == '__main__':
    main()
