# UniNet · USOD10K

用于水下显著性目标检测的UniNet项目，源于 [hao-wang1216/UniNet](https://github.com/hao-wang1216/UniNet)。当前配置为 **PVTv2-B2 / 352×352 / batch size 8**。

## 项目结构

```text
UniNet_B2_352_B8/
├── README.md
├── requirements.txt              依赖版本
├── configs/
│   ├── defaults.py               默认超参数、权重与数据路径
│   └── arguments.py              训练/测试命令行参数
├── datasets/
│   ├── augmentations.py          配对数据增强
│   ├── train_dataset.py          训练/验证数据读取
│   ├── test_dataset.py           测试数据读取
│   └── loaders.py                DataLoader
├── models/
│   ├── uninet.py                 主网络与完整连接关系
│   ├── backbones/pvt.py          PVT编码器实现
│   └── modules/
│       ├── common.py             基础卷积
│       ├── aspp.py               多尺度上下文
│       ├── attention.py          通道/空间/像素注意力
│       ├── fusion.py             MFF跨模态融合
│       ├── refinement.py         门控细化与残差解码
│       └── edges.py              Laplacian边缘与跨尺度融合
├── engine/
│   ├── trainer.py                训练调度与模型保存
│   ├── train_one_epoch.py        单轮训练
│   ├── evaluate.py               验证MAE
│   └── predictor.py              显著图推理导出
├── utils/
│   ├── losses.py                 原结构损失
│   ├── optimization.py           梯度裁剪与学习率规则
│   ├── checkpoint.py             检查点加载
│   ├── paths.py                  USOD10K目录解析
│   └── runtime.py                单卡运行环境
├── weights/pvt_v2_b2.pth         B2的ImageNet预训练权重
├── outputs/                      训练日志、检查点和预测结果
├── train.py                      训练入口
└── test.py                       测试集显著图导出入口
```

只创建有实际用途的目录；数据集继续使用服务器上的外部目录，不复制到项目中。

## 配置

集中编辑 `configs/defaults.py`：

| 配置 | 默认值 |
|---|---|
| BACKBONE | pvt_v2_b2 |
| INPUT_SIZE | 352 |
| BATCH_SIZE | 8 |
| EPOCHS | 100 |
| LEARNING_RATE | 1e-4 |
| OPTIMIZER | AdamW，weight_decay沿用1e-4 |
| DATA_ROOT | /root/autodl-tmp/data/USOD10K |

权重、日志和输出相对于项目文件的位置解析，不再依赖作者机器的绝对路径。B2需要自己的预训练权重，不能用B5改名替代。

项目已附带从 [PVT官方发布页](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth) 下载的B2权重，编码器332个参数键全部匹配。SHA-256：`80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d`。

## 环境与数据

依赖版本见 `requirements.txt`，按服务器Python 3.12 / PyTorch 2.8.0 + CUDA 12.8环境列出。RTX 5090需使用支持该显卡的CUDA版PyTorch。已有匹配环境可以直接复用；若依赖需要补齐，可在本地准备对应Linux wheel，再离线上传安装，避免服务器慢速下载。本项目不会自动联网下载权重或数据。

```text
/root/autodl-tmp/data/USOD10K/
├── USOD10K-TR/
│   ├── USOD10K-TR-RGB/
│   ├── USOD10K-TR-GT/
│   └── USOD10K-TR-depth/
├── USOD10K-Val/
│   ├── USOD10K-Val-RGB/
│   ├── USOD10K-Val-GT/
│   └── USOD10K-Val-depth/
└── USOD10K-TE/
    ├── USOD10K-TE-RGB/
    ├── USOD10K-TE-GT/
    └── USOD10K-TE-depth/
```

辅助depth图仍按源数据读取方式转换为三通道。RGB/GT/depth应已正确一一配对；本次没有添加新的样本筛选或拆分策略。

## 使用

在项目根目录执行，先查看参数：

```bash
python train.py --help
python test.py --help
```

正式训练（会占用GPU，执行前确认环境与数据已就绪）：

```bash
python -u train.py --data_root /root/autodl-tmp/data/USOD10K
```

恢复训练仍使用源代码已有的检查点加载逻辑：

```bash
python -u train.py --resume /path/to/P_best_epoch_20.pth
```

默认检查点/日志目录是 `outputs/checkpoints/`。最佳权重保存为 `D_UniNet.pth`，同时保存 `P_best_epoch_*.pth` 完整检查点。`--epoch_save` 分支沿用源实现：只打印提示，不额外写盘；实际模型由最佳MAE分支保存。

导出测试集显著图：

```bash
python -u test.py --data_root /root/autodl-tmp/data/USOD10K --pth_path outputs/checkpoints/D_UniNet.pth
```

输出位于 `outputs/predictions/USOD10K-TE/`。`test.py`保持源代码的显著图导出职责，没有新增一套评测指标；验证MAE由 `engine/evaluate.py`计算。

## 网络阅读顺序与算法位置

1. `models/uninet.py`装配整体网络。
2. RGB与depth沿batch拼成 `[2B,3,H,W]`，共享PVT编码器。
3. 四级ASPP投影为32通道；Laplacian边缘与跨尺度融合增强特征。
4. 沿batch拆回两路模态，经MFF执行跨模态融合。
5. 深/中尺度经RFD与引导门控做三轮迭代预测；第2轮起启用门控。
6. 最终预测另由浅层 `mff_4 → downsample → Fus → out_pred → final` 分支生成。

352输入的四级特征尺寸从浅到深为88、44、22、11；`x4`为最浅层，`x1`为最深层。batch size 8表示8组样本，共享编码器实际接收16张输入。

模块注释按实际实现职责标明位置。ASPP是已有通用组件；不能因代码中出现就称为作者原创。当前代码没有直接定义BFE/CFI/BFR类，未将它们与图示作未经确认的一一映射。`BasicConv2d`默认Conv+BN；门控alpha/beta是可学习标量，引导图来自解码特征投影。

## 本次整理的边界

模型结构、成员名、损失公式与权重、优化器设置、增强、shuffle、模型选择和推理后处理保持源实现。验证仍复用包含随机增强的训练数据类。学习率按初始值计算阶梯衰减，默认在第 50、100 轮开始时各降低一次。

工程变更限于目录/导入、配置集中化、入口封装、USOD10K路径接线、清理无效重复定义与历史调试注释。训练函数显式使用传入的优化器，取代原入口的同一全局对象。代码不再携带兼容模块、旧B5权重、历史字节码、旧Git元数据或归档文件。原始UniNet目录独立保留。

## 核查范围

已核查Python语法、配置/导入、332个编码器权重键、完整网络state_dict键与形状、合成配对数据读取，以及352×352的CPU前向。使用同一参数时，整理前B2参考实现与本项目的三轮阶段预测和最终预测逐项最大绝对差值均为0。结构损失数值也一致。

核查环境为Python 3.11、PyTorch 2.8.0+cu128、timm 0.9.16、einops 0.8.1，前向使用CPU。此结果不等于已在服务器完成正式训练；本次未启动服务器训练，也未更改其正在运行的实验。
