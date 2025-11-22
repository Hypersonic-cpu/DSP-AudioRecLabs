# DSP - Audio Recognition Lab

## Raw Data and Pre-processing

```
data/
├── 0/
│   ├── sample1.wav
│   └── sample2.wav
├── 1/
│   └── ...
└── ...
```

* 每个文件夹代表一个类别，内含WAV音频文件（采样率44100Hz）. 
* `src/driver/loader.py` 提供将 `.wav` 转换为 `ndarray` with sample rate 的函数. 

## Freq-Domain Classifier
Source code:
```
src
└── freq_domain
    ├── __init__.py
    ├── classifiers.py  # 分类器 wrapper 及 DTW
    ├── fourier.py      # FFT wrapper
    ├── mel.py          # Mel 特征提取, MFCC 生成
    └── train.py        # 0-9 的 MFCC Template 生成
```

-----

## - 基于时域分析

>  __DEPRECATED__： 该部分内容已经弃用, 不保证正常运行. __TODO__.

孤立词语音识别系统，使用时域特征（能量、幅度、过零率）进行分类. 

### 快速使用

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行实验（指定数据路径）
python run.py --data-dir ~/Downloads/your_voice_data

# 3. 查看结果
# 结果保存在 results/ 目录
```

### 主要功能

- **预处理**：去直流、归一化
- **端点检测**：双门限法（能量+过零率）
- **分帧加窗**：支持矩形窗、汉明窗、海宁窗
- **特征提取**：短时能量、短时平均幅度、短时过零率
- **分类器**：KNN、朴素贝叶斯、决策树、SVM、MLP神经网络

### 实验对比

1. **分类器对比**：比较不同分类器性能
2. **窗函数对比**：比较三种窗函数效果
3. **特征分析**：可视化特征分布

### 配置

修改 `config.py` 可调整参数：
- 数据路径
- 帧长/帧移
- 端点检测门限
- 分类器超参数

### 命令行参数

```bash
python run.py \
    --data-dir ~/Downloads/data \        # 数据路径（必需）
    --experiment all \                   # all/classifier/window/feature
    --window-type hamming                # rectangular/hamming/hanning
```

### 测试

```bash
# 基础功能测试
python test_basic.py
```

### 项目结构

```
DSP/
├── src/                    # 核心代码
│   ├── audio_processing.py
│   ├── feature_extraction.py
│   ├── models.py
│   └── visualization.py
├── experiments/
│   └── run_experiments.py
├── config.py              # 配置文件
├── run.py                 # 启动脚本
└── results/               # 实验结果（自动生成）
```
