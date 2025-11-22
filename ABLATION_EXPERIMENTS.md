# 消融实验使用指南

本文档说明如何使用新增的消融实验功能对比不同超参数对模型性能的影响。

## 📋 目录

1. [数据集配置](#数据集配置)
2. [核心训练模块](#核心训练模块)
3. [消融实验](#消融实验)
4. [配置文件说明](#配置文件说明)

---

## 🗂️ 数据集配置

### 数据集路径

在 `config.py` 中已经配置好两个数据集的路径：

```python
DATASET_PATHS = {
    'name': '~/Downloads/speech_data_name',     # 中文名字数据集
    'number': '~/Downloads/speech_data_number', # 数字数据集
}
```

### 切换数据集

**方法1：修改配置文件**

在 `config.py` 中修改 `DATASET_TYPE`：

```python
DATASET_TYPE = 'name'   # 使用中文名字数据集
# 或
DATASET_TYPE = 'number' # 使用数字数据集
```

**方法2：使用环境变量**

```bash
# 使用中文名字数据集
export DATASET_TYPE=name
python ablation_study.py

# 使用数字数据集
export DATASET_TYPE=number
python ablation_study.py
```

**方法3：使用命令行参数**

```bash
# 使用中文名字数据集
python ablation_study.py --dataset name

# 使用数字数据集
python ablation_study.py --dataset number
```

---

## 🎯 核心训练模块

新增的 `train_model.py` 提供了简化的训练接口。

### 快速训练

```python
from train_model import quick_experiment

# 使用默认参数训练
results = quick_experiment()

# 修改分类器
results = quick_experiment(classifier_type='svm')

# 修改学习率
results = quick_experiment(classifier_type='mlp', learning_rate=0.01)

# 修改帧长和帧移
results = quick_experiment(frame_length_ms=25, frame_shift_ms=15)
```

### 分步操作

```python
from train_model import load_dataset, train_and_evaluate

# 1. 加载数据
X, y, class_names, feature_names = load_dataset(
    data_dir='~/Downloads/speech_data_name',
    frame_length_ms=20,
    frame_shift_ms=10
)

# 2. 训练和评估
results = train_and_evaluate(
    X, y,
    classifier_type='mlp',
    learning_rate=0.005
)

print(f"准确率: {results['accuracy']:.4f}")
```

---

## 🔬 消融实验

### 1. 学习率对比实验

对比不同学习率对MLP模型性能的影响。

**运行命令：**

```bash
# 在中文名字数据集上测试
python ablation_study.py --experiment lr --dataset name

# 在数字数据集上测试
python ablation_study.py --experiment lr --dataset number
```

**配置学习率范围：**

在 `config.py` 中修改：

```python
LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
```

**输出：**
- 准确率对比图表：`results/ablation_learning_rate/learning_rate_comparison.png`
- 详细结果：`results/ablation_learning_rate/results.json`
- 文本摘要：`results/ablation_learning_rate/results_summary.txt`

---

### 2. 帧长对比实验

对比不同帧长（采样点个数）对模型性能的影响。

**运行命令：**

```bash
# 在中文名字数据集上测试
python ablation_study.py --experiment frame_length --dataset name

# 在数字数据集上测试
python ablation_study.py --experiment frame_length --dataset number

# 使用SVM分类器
python ablation_study.py --experiment frame_length --classifier svm
```

**配置帧长范围：**

在 `config.py` 中修改：

```python
FRAME_LENGTH_MS_RANGE = [10, 15, 20, 25, 30, 40]  # 毫秒
```

**输出：**
- 准确率对比图表：`results/ablation_frame_length/frame_length_comparison.png`
- 详细结果：`results/ablation_frame_length/results.json`

---

### 3. 帧移对比实验

对比不同帧移对模型性能的影响。

**运行命令：**

```bash
# 在中文名字数据集上测试
python ablation_study.py --experiment frame_shift --dataset name

# 在数字数据集上测试
python ablation_study.py --experiment frame_shift --dataset number
```

**配置帧移范围：**

在 `config.py` 中修改：

```python
FRAME_SHIFT_MS_RANGE = [5, 10, 15, 20, 25]  # 毫秒
```

**输出：**
- 准确率对比图表：`results/ablation_frame_shift/frame_shift_comparison.png`
- 详细结果：`results/ablation_frame_shift/results.json`

---

### 4. 运行所有消融实验

```bash
# 在中文名字数据集上运行所有实验
python ablation_study.py --experiment all --dataset name

# 在数字数据集上运行所有实验
python ablation_study.py --experiment all --dataset number
```

---

## ⚙️ 配置文件说明

### config.py 主要参数

```python
# ==================== 数据集配置 ====================
DATASET_TYPE = 'name'  # 'name' 或 'number'

DATASET_PATHS = {
    'name': '~/Downloads/speech_data_name',
    'number': '~/Downloads/speech_data_number',
}

# ==================== 音频处理参数 ====================
SAMPLE_RATE = 44100         # 采样率
FRAME_LENGTH_MS = 20        # 默认帧长（毫秒）
FRAME_SHIFT_MS = 10         # 默认帧移（毫秒）

# ==================== MLP参数 ====================
MLP_HIDDEN_LAYERS = [64, 64, 32]
MLP_LEARNING_RATE = 0.005
MLP_EPOCHS = 1000
MLP_BATCH_SIZE = 108

# ==================== 消融实验参数 ====================
LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
FRAME_LENGTH_MS_RANGE = [10, 15, 20, 25, 30, 40]
FRAME_SHIFT_MS_RANGE = [5, 10, 15, 20, 25]
```

---

## 📊 结果解读

### 准确率对比图

消融实验会生成两条曲线：
- **蓝色线**：训练集准确率
- **红色线**：测试集准确率
- **绿色星标**：最佳参数点

### 结果文件

每个实验会生成以下文件：

1. **PNG图表**：可视化对比结果
2. **JSON文件**：完整的实验数据，方便后续分析
3. **TXT摘要**：易读的文本格式结果

### 示例结果摘要

```
======================================================================
学习率对比
======================================================================
数据集: speech_data_name
类别: 张三, 李四, 王五, ...
参数: learning_rate
时间: 2025-10-26 10:30:00

参数值          训练准确率      测试准确率
----------------------------------------------------------------------
0.0001          0.9200          0.8800
0.0005          0.9500          0.9100
0.001           0.9700          0.9300
0.005           0.9900          0.9500  <-- 最佳
0.01            0.9950          0.9200
0.05            0.9990          0.8500

======================================================================
最佳参数: 0.005
最佳准确率: 0.9500
======================================================================
```

---

## 🚀 快速开始示例

### 完整工作流程

```bash
# 1. 查看可用数据集
ls ~/Downloads/speech_data_*

# 2. 快速测试（使用默认参数）
python train_model.py

# 3. 在中文名字数据集上运行所有消融实验
python ablation_study.py --experiment all --dataset name --classifier mlp

# 4. 在数字数据集上运行所有消融实验
python ablation_study.py --experiment all --dataset number --classifier mlp

# 5. 查看结果
ls results/ablation_*
```

---

## 💡 提示

1. **中文显示**：代码已配置支持中文字体显示，图表中的中文名字会正确显示
2. **实验时间**：完整的消融实验可能需要较长时间，建议先用单个实验测试
3. **结果目录**：所有结果保存在 `results/` 目录下，按实验类型分类
4. **自定义参数**：可以在 `config.py` 中调整参数范围以适应你的需求

---

## 📝 注意事项

1. 确保数据集路径正确
2. 学习率实验仅适用于MLP分类器
3. 帧长和帧移实验会重新加载数据，因此耗时较长
4. 建议先在小数据集上测试，确认流程正确后再运行完整实验
