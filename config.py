"""
配置文件：所有超参数和路径配置
"""
import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集路径配置
# 方式1：使用项目内的data目录（默认）
# DATA_DIR = os.path.join(BASE_DIR, 'data')

# 方式2：使用downloads目录（推荐，如果数据在downloads文件夹）
# DATA_DIR = os.path.join(os.path.expanduser('~'), 'Downloads', 'speech_data')

# 方式3：使用自定义绝对路径
# DATA_DIR = '/path/to/your/data'

# 当前配置：优先使用环境变量，否则使用默认data目录
DATA_DIR = os.environ.get('SPEECH_DATA_DIR', r'/Users/ding/Downloads/speech_data_cleaned')

# 结果保存目录
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== 音频参数 ====================
SAMPLE_RATE = 44100              # 采样率（Hz）

# ==================== 预处理参数 ====================
NORMALIZE = True                 # 是否归一化

# ==================== 端点检测参数 ====================
FRAME_LENGTH_MS = 20             # 帧长（毫秒）
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
MLP_HIDDEN_LAYERS = [128, 64, 64]
MLP_LEARNING_RATE = 0.005
MLP_EPOCHS = 2000
MLP_BATCH_SIZE = 108

# ==================== 实验参数 ====================
TEST_SIZE = 0.2                  # 测试集比例
RANDOM_SEED = 42                 # 随机种子

# ==================== 可视化参数 ====================
FIGURE_DPI = 150
FIGURE_SIZE = (12, 8)
