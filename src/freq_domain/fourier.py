import torch
import numpy as np
import math
import sys

# --- 尝试导入自定义 FFT 扩展 ---
try:
    # 假设你的编译产物名为 custom_fft 且扩展名为 _C
    import custom_fft._C as custom_ops
    HAS_CUSTOM_FFT = True
except ImportError:
    print("Warning: 'custom_fft' extension not found. Using torch.fft as fallback for debugging.")
    HAS_CUSTOM_FFT = False

def pre_emphasis(signal, coeff=0.97):
    """ 预加重: y[n] = x[n] - a*x[n-1] """
    return torch.cat((signal[0:1], signal[1:] - coeff * signal[:-1]))

def fft_wrapper(frames, n_fft):
    """
    (B-3) 调用自定义的 Metal/OpenMP FFT 算子
    输入: frames [Batch_Size, n_fft] (实数, float32)
    输出: magnitude [Batch_Size, n_fft/2 + 1]
    """
    batch_size, length = frames.shape
    
    # 1. 数据类型转换: Real -> Complex
    # 你的 sample.py 显示输入是 torch.cfloat，所以我们将实数转为虚部为0的复数
    # view_as_complex 需要最后一维是2，这里直接构造复数张量更安全
    complex_frames = torch.complex(frames, torch.zeros_like(frames))
    
    if HAS_CUSTOM_FFT:
        # 根据设备选择调用 Metal 或 CPU 版本
        # 注意: 你的自定义算子需要支持传入 Tensor 所在的 device，或者你需要先转 device
        if frames.device.type == 'mps':
            # 假设 metal_fft 是在 GPU 上执行
            output_complex = custom_ops.metal_fft(complex_frames)
        else:
            # CPU (OpenMP)
            output_complex = custom_ops.cpu_fft(complex_frames.cpu())
            if frames.device.type != 'cpu':
                output_complex = output_complex.to(frames.device)
    else:
        # Fallback: 如果没编译好扩展，用 torch 原生 fft 兜底，保证代码可运行
        output_complex = torch.fft.fft(complex_frames)

    # 2. 计算幅值 |X(k)|
    magnitude = torch.abs(output_complex)
    
    # 3. 截取前一半 (N/2 + 1)，因为实数信号频谱共轭对称
    # 假设自定义 FFT 返回的是全长 N
    cutoff = int(n_fft / 2 + 1)
    return magnitude[:, :cutoff]

def stft_custom(signal, sample_rate, frame_len_ms=25, frame_shift_ms=10, fft_size=512):
    """
    实现: 预加重 -> 分帧 -> 加窗 -> Custom FFT -> 能量谱
    """
    # 确保信号是 Tensor
    if not torch.is_tensor(signal):
        signal = torch.from_numpy(signal).float()
    
    signal_len = len(signal)
    frame_len = int(sample_rate * frame_len_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)
    
    # 1. 预加重
    emphasized_signal = pre_emphasis(signal)
    
    # 2. 分帧 (使用 unfold 也可以，这里用索引切片逻辑，清晰直观)
    num_frames = int(np.ceil(float(np.abs(signal_len - frame_len)) / frame_shift))
    
    # Padding
    pad_len = (num_frames - 1) * frame_shift + frame_len - signal_len
    if pad_len > 0:
        emphasized_signal = torch.cat((emphasized_signal, torch.zeros(pad_len).to(signal.device)))
        
    # 构建帧索引矩阵 [num_frames, frame_len]
    indices = torch.arange(0, frame_len).unsqueeze(0).to(signal.device) + \
              torch.arange(0, num_frames * frame_shift, frame_shift).unsqueeze(1).to(signal.device)
    
    frames = emphasized_signal[indices.long()]
    
    # 如果帧长小于 fft_size，需要补零 (Zero Padding)
    if frame_len < fft_size:
        padding = torch.zeros((num_frames, fft_size - frame_len)).to(signal.device)
        # 注意：通常先加窗再补零，或者补零对加窗无影响（如果是尾部补零）
        # 这里为了简单，我们先把 frames 扩展出来，但窗函数只作用于前 frame_len
        frames_padded = torch.cat((frames, padding), dim=1)
    else:
        frames_padded = frames[:, :fft_size] # 截断

    # 3. 加窗 (Hamming)
    # w[n] = 0.54 - 0.46 * cos(...)
    n = torch.arange(0, frame_len).to(signal.device)
    window = 0.54 - 0.46 * torch.cos((2 * math.pi * n) / (frame_len - 1))
    
    # 广播乘法: 只乘以前 frame_len 部分
    frames_padded[:, :frame_len] *= window
    
    # 4. 调用 FFT Wrapper
    mag_frames = fft_wrapper(frames_padded, fft_size)
    
    # 5. 计算能量谱 Power Spectrum: |X(k)|^2 / N
    pow_frames = (mag_frames ** 2) / fft_size
    
    return pow_frames