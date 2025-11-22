import numpy as np
import torch
from fourier import stft_custom
from mel import mel_wrapper

def dtw_distance(s1, s2):
    """
    (B-2) DTW 算法
    输入: 两个 MFCC 序列 [Frames, Feature_Dim]
    s1: 测试输入 (N 帧)
    s2: 参考模板 (M 帧)
    """
    # 确保都在同一设备
    if s1.device != s2.device:
        s2 = s2.to(s1.device)

    n, dim = s1.shape
    m, _ = s2.shape
    
    # 1. 计算欧式距离矩阵 Distance Matrix
    # 利用广播: (N, 1, D) - (1, M, D) -> (N, M, D) -> norm -> (N, M)
    dist = torch.norm(s1[:, None, :] - s2[None, :, :], p=2, dim=2)
    
    # 2. 初始化累计距离矩阵 D
    # 多加一行一列用于处理边界条件
    dtw_mat = torch.full((n + 1, m + 1), float('inf')).to(s1.device)
    dtw_mat[0, 0] = 0
    
    # 3. 动态规划 (迭代填表)
    # Python 循环较慢，但对于孤立词识别(帧数少)通常可接受
    # 截图公式: D(i,j) = d(i,j) + min(D(i-1,j), D(i-1,j-1), D(i,j-1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist[i-1, j-1]
            # 寻找前序路径的最小值
            min_prev = torch.min(torch.stack([
                dtw_mat[i-1, j],   # 插入
                dtw_mat[i, j-1],   # 删除
                dtw_mat[i-1, j-1]  # 匹配
            ]))
            dtw_mat[i, j] = cost + min_prev
            
    # 返回归一化距离或总距离，这里返回总距离 D(N,M)
    return dtw_mat[n, m]

def recognition_wrapper(audio_data, sample_rate, classifier_func, templates):
    """
    (B-1) 识别框架
    
    参数:
      audio_data: np.ndarray (1D)
      classifier_func: dtw_distance 函数
      templates: dict {'0': mfcc_tensor, '1': ...}
    """
    # 转 Tensor (如果在 Mac 上有 M1/M2，推荐用 mps)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    if not torch.is_tensor(audio_data):
        signal = torch.from_numpy(audio_data).float().to(device)
    else:
        signal = audio_data.float().to(device)
        
    # --- 特征提取 ---
    # 1. STFT (包含调用 Custom FFT)
    pow_frames = stft_custom(signal, sample_rate)
    
    # 2. MFCC (包含 Mel 滤波和 DCT)
    # 实验要求通常取 13 阶 MFCC
    input_mfcc = mel_wrapper(pow_frames, sample_rate, n_ceps=13)
    
    # --- 模式匹配 ---
    best_digit = None
    min_dist = float('inf')
    
    print(f"Input MFCC shape: {input_mfcc.shape}")
    
    for digit, ref_mfcc in templates.items():
        # 确保参考模板也是 Tensor 且在设备上
        if not torch.is_tensor(ref_mfcc):
            ref_mfcc = torch.from_numpy(ref_mfcc).float()
        ref_mfcc = ref_mfcc.to(device)
        
        # 计算 DTW 距离
        d = classifier_func(input_mfcc, ref_mfcc)
        
        print(f"Distance to template '{digit}': {d:.4f}")
        
        if d < min_dist:
            min_dist = d
            best_digit = digit
            
    return best_digit
