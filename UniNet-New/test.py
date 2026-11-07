# 测试集显著图导出入口；当前脚本不额外增加指标计算公式。
from configs.arguments import parse_test_args
from utils.runtime import configure_cuda_environment

def main():
    opt = parse_test_args()
    configure_cuda_environment()
    from engine.predictor import run_inference
    run_inference(opt)

if __name__ == '__main__':
    main()
