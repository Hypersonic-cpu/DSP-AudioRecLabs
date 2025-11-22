import os
import torch
import numpy as np
import glob
from pathlib import Path

# 引入之前的模块 (假设它们在同一目录下)
from fourier import stft_custom
from mel import mel_wrapper

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
# 路径配置
# 信号处理参数
SAMPLE_RATE = 8000      # 假设采样率为 8k, 具体由 load 函数决定
N_FFT = 512             # FFT 点数
N_MFCC = 13             # MFCC 特征维度 (实验要求 12-16)
FRAME_LEN_MS = 25       # 帧长 25ms
FRAME_SHIFT_MS = 10     # 帧移 10ms

# 设备配置
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==========================================
# 2. Helper Functions
# ==========================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 模拟题目要求的加载函数 (实际运行时请确保环境中有此函数或替换为真实实现)
def my_default_load(path: str) -> tuple[np.ndarray, int]:
    """
    读取音频 array 和采样率.
    这里是一个 Mock 实现，实际使用时请替换为你自己的读取库 (如 scipy.io.wavfile 或 librosa)
    """
    try:
        # 尝试使用 scipy 读取 (如果存在)
        import scipy.io.wavfile as wav
        sr, data = wav.read(path)
        # 归一化到 [-1, 1] 并转 float
        if data.dtype == np.int16:
            data = data / 32768.0
        return data.astype(np.float32), sr
    except ImportError:
        # 如果没有库，生成随机噪声用于代码跑通测试
        # print(f"[Mock] Loading {path} with random noise.")
        return np.random.uniform(-0.5, 0.5, 16000), 8000

def process_single_file(filepath, device=DEVICE):
    """
    处理单个音频文件: Load -> STFT -> Mel -> MFCC
    """
    # 1. 加载音频
    audio_arr, sr = my_default_load(filepath)
    
    # 2. 转 Tensor 并移至设备
    # 确保输入是 1D Tensor
    signal = torch.from_numpy(audio_arr).float().to(device)
    if signal.ndim > 1:
        signal = signal.mean(dim=1) # 混音为单声道
        
    # 3. STFT (fourier.py)
    # 注意: stft_custom 内部调用了 fft_wrapper (你的 Metal/OpenMP FFT)
    pow_frames = stft_custom(signal, sr, 
                             frame_len_ms=FRAME_LEN_MS, 
                             frame_shift_ms=FRAME_SHIFT_MS, 
                             fft_size=N_FFT)
    
    # 4. Mel Feature Extraction (mel.py)
    mfcc_features = mel_wrapper(pow_frames, sr, 
                                n_fft=N_FFT, 
                                n_ceps=N_MFCC)
    
    return mfcc_features

# ==========================================
# 3. Train Wrapper (Main Logic)
# ==========================================

def train_wrapper(data_root_path, output_dir, save_model_path):
    """
    执行特征提取并构建模板库。
    
    Args:
        data_root_path: 源数据根目录 (包含 0-9 文件夹)
        output_dir: MFCC 文件输出目录
        save_model_path: 最终模板字典的保存路径 (.pt)
    """
    ensure_dir(output_dir)
    
    # 用于存储最终的模板库 (Model)
    # 结构: {'0': mfcc_tensor, '1': mfcc_tensor, ...}
    templates_library = {}
    
    print(f"Start processing data from: {data_root_path}")
    print(f"Device: {DEVICE}")
    
    # 遍历数字 0 到 9
    for digit in range(10):
        digit_str = str(digit)
        src_dir = os.path.join(data_root_path, digit_str)
        
        if not os.path.exists(src_dir):
            print(f"Warning: Directory for digit '{digit}' not found at {src_dir}, skipping.")
            continue
            
        # 获取该数字下的所有音频文件 (假设是 wav)
        # 支持 .wav, .mp3, .flac 等
        files = []
        for ext in ['*.wav', '*.mp3']:
            files.extend(glob.glob(os.path.join(src_dir, ext)))
            
        if not files:
            print(f"Warning: No audio files found in {src_dir}")
            continue
            
        print(f"Processing digit '{digit}': found {len(files)} samples.")
        
        # 遍历该数字下的所有样本
        digit_mfccs = []
        for i, file_path in enumerate(files):
            try:
                # --- 核心处理 ---
                mfcc_tensor = process_single_file(file_path, device=DEVICE)
                
                # 保存独立的 MFCC 文件 (可选，便于调试或后续高级训练)
                filename = Path(file_path).stem
                save_name = f"{digit}_{filename}.pt"
                save_full_path = os.path.join(output_dir, save_name)
                torch.save(mfcc_tensor.cpu(), save_full_path) # 保存到 CPU 以便兼容
                
                digit_mfccs.append(mfcc_tensor)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # --- 模板选择策略 (Template Selection) ---
        # 策略: 简单起见，选取该数字的第一个样本作为"参考模板"
        # (在更复杂的系统中，这里可以做 K-Means 聚类或者计算所有样本的中心)
        if len(digit_mfccs) > 0:
            # 这里我们选择列表中的第一个作为代表
            representative_mfcc = digit_mfccs[0]
            templates_library[digit_str] = representative_mfcc.cpu()
            print(f"  -> Saved {len(digit_mfccs)} MFCCs. Selected sample 0 as template for '{digit}'.")
        else:
            print(f"  -> No valid MFCCs generated for '{digit}'.")

    # 5. 保存最终的模型文件 (模板库)
    if templates_library:
        torch.save(templates_library, save_model_path)
        print("="*40)
        print(f"Training complete.")
        print(f"Templates library saved to: {save_model_path}")
        print(f"Model contains keys: {list(templates_library.keys())}")
    else:
        print("Error: No templates were generated. Check data path.")
