"""
配置文件：所有超参数和路径配置
"""
import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== 数据集路径配置 ====================
# 数据集类型选择：'name' 或 'number'
# - 'name': 中文名字数据集
# - 'number': 数字数据集
DATASET_TYPE = os.environ.get('DATASET_TYPE', 'name')  # 默认使用中文名字数据集

# 数据集路径字典
DATASET_PATHS = {
    'name': os.path.join(os.path.expanduser('~'), 'Downloads', 'speech_data_name'),     # 中文名字
    'number': os.path.join(os.path.expanduser('~'), 'Downloads', 'speech_data_number'), # 数字
}

# 当前使用的数据集路径（可通过环境变量 SPEECH_DATA_DIR 覆盖）
DATA_DIR = os.environ.get('SPEECH_DATA_DIR', DATASET_PATHS[DATASET_TYPE])

# 结果保存目录
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== 音频参数 ====================
SAMPLE_RATE = 44100              # 采样率（Hz）

# ==================== 预处理参数 ====================
NORMALIZE = True                 # 是否归一化

# ==================== 端点检测参数 ====================
FRAME_LENGTH_MS = 25             # 帧长（毫秒）
FRAME_SHIFT_MS = 10              # 帧移（毫秒）

# 自动计算帧长和帧移（采样点数）
FRAME_LENGTH = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)
FRAME_SHIFT = int(SAMPLE_RATE * FRAME_SHIFT_MS / 1000)

# 双门限参数
ENERGY_HIGH_RATIO = 0.5          # 能量高门限（相对于最大能量的比例）
ENERGY_LOW_RATIO = 0.1           # 能量低门限
ZCR_THRESHOLD_RATIO = 1.5        # 过零率门限（相对于平均过零率的倍数）

# ==================== 窗函数类型 ====================
WINDOW_TYPES = ['rectangular', 'hamming', 'hanning']

# ==================== 特征提取参数 ====================
# 对每个特征序列提取的统计量
FEATURE_STATS = ['mean', 'std', 'max', 'min', 'median']

# ==================== 分类器参数 ====================
# KNN
KNN_N_NEIGHBORS = 3

# SVM
SVM_C = 1.0
SVM_KERNEL = 'rbf'

# MLP
MLP_HIDDEN_LAYERS = [64, 64, 32]
MLP_LEARNING_RATE = 0.005
MLP_EPOCHS = 1000
MLP_BATCH_SIZE = 108

# ==================== 实验参数 ====================
TEST_SIZE = 0.2                  # 测试集比例
RANDOM_SEED = 42                 # 随机种子

# ==================== 可视化参数 ====================
FIGURE_DPI = 150
FIGURE_SIZE = (12, 8)

# ==================== 消融实验参数 ====================
# 学习率sweep范围（增加更多点让曲线平滑）
LEARNING_RATES = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08]

# 帧长sweep范围（毫秒）- 增加更多点
FRAME_LENGTH_MS_RANGE = [8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50]

# 帧移sweep范围（毫秒）- 扩大范围，一般最优在帧长的30-50%
# 对于20ms帧长，最优通常在6-10ms之间，但我们测试更大范围
FRAME_SHIFT_MS_RANGE = [3, 5, 7, 8, 10, 12, 15, 18, 20, 25, 30]
