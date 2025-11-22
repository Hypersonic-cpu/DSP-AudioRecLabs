# 快速开始指南

## 📦 安装依赖

首先确保安装所有必要的Python包：

```bash
cd /Users/ding/Desktop/DesktopAir/DSP/DSP-TimeDomainAudioRec
pip install -r requirements.txt
```

**注意**：如果PyTorch安装失败，请使用以下命令：

```bash
# macOS (Apple Silicon)
pip install torch torchvision torchaudio

# macOS (Intel)
pip install torch torchvision torchaudio

# 或者使用conda
conda install pytorch torchvision torchaudio -c pytorch
```

## 🎯 主要功能

### 1. 数据集切换

项目支持两个数据集：
- **中文名字数据集**：`~/Downloads/speech_data_name`
- **数字数据集**：`~/Downloads/speech_data_number`

**切换方法**：

在 [config.py](config.py#L13) 中修改：

```python
DATASET_TYPE = 'name'   # 中文名字
# 或
DATASET_TYPE = 'number' # 数字
```

### 2. 快速训练

使用新的核心训练模块 [train_model.py](train_model.py) 快速训练模型：

```bash
# 使用默认参数训练MLP
python train_model.py

# 输出示例：
# 准确率: 0.9500
```

### 3. 运行消融实验

#### 学习率对比

对比不同学习率对MLP性能的影响：

```bash
# 在中文名字数据集上测试
python ablation_study.py --experiment lr --dataset name

# 在数字数据集上测试
python ablation_study.py --experiment lr --dataset number
```

**结果位置**：`results/ablation_learning_rate/`

#### 帧长对比

对比不同帧长（采样窗口大小）对性能的影响：

```bash
python ablation_study.py --experiment frame_length --dataset name
```

**结果位置**：`results/ablation_frame_length/`

#### 帧移对比

对比不同帧移（帧之间的重叠程度）对性能的影响：

```bash
python ablation_study.py --experiment frame_shift --dataset name
```

**结果位置**：`results/ablation_frame_shift/`

#### 运行所有实验

```bash
# 在中文名字数据集上运行所有消融实验
python ablation_study.py --experiment all --dataset name

# 在数字数据集上运行所有消融实验
python ablation_study.py --experiment all --dataset number
```

### 4. 原有功能（完整实验流程）

运行原有的完整实验流程：

```bash
# 运行所有实验
python run.py --experiment all

# 只运行分类器对比
python run.py --experiment classifier

# 只运行窗函数对比
python run.py --experiment window

# 可视化样本
python run.py --experiment visualize
```

## 📊 结果查看

所有实验结果保存在 `results/` 目录下：

```
results/
├── ablation_learning_rate/      # 学习率消融实验
│   ├── learning_rate_comparison.png
│   ├── results.json
│   └── results_summary.txt
├── ablation_frame_length/       # 帧长消融实验
│   ├── frame_length_comparison.png
│   ├── results.json
│   └── results_summary.txt
├── ablation_frame_shift/        # 帧移消融实验
│   ├── frame_shift_comparison.png
│   ├── results.json
│   └── results_summary.txt
├── exp1_classifier_comparison/  # 分类器对比
├── exp2_window_comparison/      # 窗函数对比
└── exp3_feature_analysis/       # 特征分析
```

## 🎨 中文显示

代码已经配置好中文字体支持，图表中的中文名字会正确显示。

如果遇到中文显示问题，请检查：

1. matplotlib的字体配置
2. 系统中是否安装了中文字体（macOS自带）

可视化模块 [src/visualization.py](src/visualization.py) 会自动检测并使用以下字体：
- Arial Unicode MS（macOS）
- PingFang SC（苹方）
- Heiti SC（黑体-简）
- 等等

## ⚙️ 参数配置

### 主要参数

在 [config.py](config.py) 中可以修改：

```python
# 数据集选择
DATASET_TYPE = 'name'  # 'name' 或 'number'

# 音频处理参数
FRAME_LENGTH_MS = 20   # 帧长（毫秒）
FRAME_SHIFT_MS = 10    # 帧移（毫秒）

# MLP训练参数
MLP_LEARNING_RATE = 0.005
MLP_EPOCHS = 1000
MLP_BATCH_SIZE = 108

# 消融实验参数
LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
FRAME_LENGTH_MS_RANGE = [10, 15, 20, 25, 30, 40]
FRAME_SHIFT_MS_RANGE = [5, 10, 15, 20, 25]
```

## 💡 代码结构

```
DSP-TimeDomainAudioRec/
├── config.py                    # 配置文件（修改参数）
├── train_model.py               # 核心训练模块（新增）
├── ablation_study.py            # 消融实验脚本（新增）
├── run.py                       # 原有运行脚本
├── src/
│   ├── models.py               # 分类器模型
│   ├── audio_processing.py     # 音频处理
│   ├── feature_extraction.py   # 特征提取
│   └── visualization.py        # 可视化（支持中文）
└── experiments/
    └── run_experiments.py      # 完整实验流程
```

## 🔍 示例工作流

### 场景1：对比两个数据集的性能

```bash
# 1. 在中文名字数据集上训练
python ablation_study.py --experiment all --dataset name

# 2. 在数字数据集上训练
python ablation_study.py --experiment all --dataset number

# 3. 对比结果
# 查看 results/ablation_*/results_summary.txt
```

### 场景2：找到最佳学习率

```bash
# 运行学习率消融实验
python ablation_study.py --experiment lr --dataset name

# 查看结果
cat results/ablation_learning_rate/results_summary.txt

# 输出会显示最佳学习率，例如：
# 最佳参数: 0.005
# 最佳准确率: 0.9500
```

### 场景3：快速测试不同参数

使用Python交互式环境：

```python
from train_model import quick_experiment

# 测试不同学习率
for lr in [0.001, 0.005, 0.01]:
    result = quick_experiment(learning_rate=lr, verbose=False)
    print(f"LR={lr}: Accuracy={result['accuracy']:.4f}")

# 测试不同帧长
for frame_len in [15, 20, 25]:
    result = quick_experiment(frame_length_ms=frame_len, verbose=False)
    print(f"Frame={frame_len}ms: Accuracy={result['accuracy']:.4f}")
```

## ❓ 常见问题

### Q1: 如何只运行一个简单的测试？

```bash
python train_model.py
```

这会使用默认参数快速训练一个MLP模型。

### Q2: 消融实验需要多长时间？

- 学习率实验：约10-30分钟（取决于样本数）
- 帧长实验：约30-60分钟（需要重新提取特征）
- 帧移实验：约30-60分钟（需要重新提取特征）

### Q3: 如何自定义消融实验的参数范围？

在 [config.py](config.py#L77-L84) 中修改：

```python
LEARNING_RATES = [0.001, 0.005, 0.01]  # 自定义学习率范围
FRAME_LENGTH_MS_RANGE = [15, 20, 25]   # 自定义帧长范围
```

### Q4: 如何使用不同的分类器进行消融实验？

```bash
# 使用SVM
python ablation_study.py --experiment frame_length --classifier svm

# 使用KNN
python ablation_study.py --experiment frame_shift --classifier knn
```

**注意**：学习率实验仅适用于MLP。

## 📚 更多文档

- [ABLATION_EXPERIMENTS.md](ABLATION_EXPERIMENTS.md) - 详细的消融实验说明
- [README.md](README.md) - 项目总体介绍

## 🚀 立即开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 快速测试
python train_model.py

# 3. 运行消融实验
python ablation_study.py --experiment all --dataset name

# 4. 查看结果
ls -la results/ablation_*
```

祝实验顺利！🎉
