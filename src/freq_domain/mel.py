import torch
import numpy as np
import math

def hz_to_mel(freq):
    """ 公式: 2595 * lg(1 + f/700) """
    return 2595 * np.log10(1 + freq / 700.0)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595.0) - 1)

def get_filter_banks(n_filters, n_fft, sample_rate, device='cpu'):
    """
    (B-4) 手动构建三角滤波器组 H_m(k)
    """
    low_freq_mel = hz_to_mel(0)
    high_freq_mel = hz_to_mel(sample_rate / 2)
    
    # Mel 刻度上均匀分布点
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    
    # 映射到 FFT bin 索引
    bin = np.floor((n_fft + 1) * hz_points / sample_rate)
    
    fbank = torch.zeros((n_filters, int(n_fft / 2 + 1))).to(device)
    
    for m in range(1, n_filters + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
            
    return fbank

def dct_custom(log_mel_energy, num_ceps=12):
    """
    离散余弦变换 (DCT-II)
    """
    num_frames, n_filters = log_mel_energy.shape
    device = log_mel_energy.device
    
    # 创建 DCT 矩阵 [Filters, Ceps]
    dct_mat = torch.zeros((n_filters, num_ceps)).to(device)
    scale = math.sqrt(2.0 / n_filters)
    
    for n in range(num_ceps):
        for m in range(n_filters):
            # 截图公式: cos( pi * n * (2m + 1) / 2M )
            angle = (math.pi * n * (2 * m + 1)) / (2 * n_filters)
            dct_mat[m, n] = scale * math.cos(angle)
            
    # [Frames, Filters] @ [Filters, Ceps]
    return torch.matmul(log_mel_energy, dct_mat)

def mel_wrapper(pow_frames, sample_rate, n_fft=512, n_filt=26, n_ceps=13):
    """
    输入: 能量谱 pow_frames [Frames, N/2+1]
    输出: MFCC [Frames, n_ceps]
    """
    device = pow_frames.device
    
    # 1. 滤波器组
    fbank = get_filter_banks(n_filt, n_fft, sample_rate, device=device)
    
    # 2. 滤波能量: S(i, m)
    # [Frames, FFT_Bins] @ [Filters, FFT_Bins].T -> [Frames, Filters]
    filter_energy = torch.matmul(pow_frames, fbank.T)
    
    # 3. 取对数 (避免 log(0))
    log_energy = torch.log(filter_energy + 1e-10)
    
    # 4. DCT 得到 MFCC
    mfcc = dct_custom(log_energy, num_ceps=n_ceps)
    
    return mfcc