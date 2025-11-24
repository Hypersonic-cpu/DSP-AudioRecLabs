"""
音频处理模块：加载、预处理、端点检测、分帧加窗
"""
import numpy as np
import wave
from src.driver.loader import load_wav
# from scipy import signal as sp_signal


def remove_dc(audio_data):
    """
    去除直流分量

    Args:
        audio_data: 音频数据

    Returns:
        去直流后的音频数据
    """
    return audio_data - np.mean(audio_data)


def normalize_audio(audio_data):
    """
    归一化音频到[-1, 1]

    Args:
        audio_data: 音频数据

    Returns:
        归一化后的音频数据
    """
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        return audio_data / max_val
    return audio_data


def preprocess(audio_data):
    """
    预处理：去直流 + 归一化

    Args:
        audio_data: 原始音频数据

    Returns:
        预处理后的音频数据
    """
    audio_data = remove_dc(audio_data)
    audio_data = normalize_audio(audio_data)
    return audio_data


def compute_short_time_energy(frame):
    """
    计算短时能量

    Args:
        frame: 音频帧

    Returns:
        短时能量
    """
    return np.sum(frame ** 2)


def compute_short_time_magnitude(frame):
    """
    计算短时平均幅度

    Args:
        frame: 音频帧

    Returns:
        短时平均幅度
    """
    return np.sum(np.abs(frame))


def compute_zero_crossing_rate(frame):
    """
    计算短时过零率

    Args:
        frame: 音频帧

    Returns:
        短时过零率
    """
    signs = np.sign(frame)
    signs[signs == 0] = -1  # 将0视为负数
    zcr = np.sum(np.abs(np.diff(signs))) / 2
    return zcr


def endpoint_detection(audio_data, frame_length, frame_shift,
                      energy_high_ratio=0.5, energy_low_ratio=0.1,
                      zcr_threshold_ratio=1.5):
    """
    改进的双门限端点检测算法

    基于标准的双门限法：
    1. 计算短时能量和短时过零率
    2. 使用高门限（基于语音能量轮廓）初判
    3. 使用低门限（基于背景噪声）精确定位，寻找第一次与门限相交的点
    4. 使用过零率进一步修正起止点

    Args:
        audio_data: 音频数据
        frame_length: 帧长（采样点数）
        frame_shift: 帧移（采样点数）
        energy_high_ratio: 能量高门限比例（相对于语音能量轮廓）
        energy_low_ratio: 能量低门限比例（相对于背景噪声）
        zcr_threshold_ratio: 过零率门限比例

    Returns:
        start_point: 起始点索引
        end_point: 终止点索引
        energy_list: 每帧能量列表
        zcr_list: 每帧过零率列表
    """
    # 分帧数量不足时直接返回整体音频
    if len(audio_data) < frame_length:
        return 0, len(audio_data), np.array([]), np.array([])

    # 分帧
    n_frames = (len(audio_data) - frame_length) // frame_shift + 1

    energy_list = []
    zcr_list = []

    # 步骤1：计算每帧的短时能量和短时过零率
    for i in range(n_frames):
        start = i * frame_shift
        end = start + frame_length
        frame = audio_data[start:end]

        energy = compute_short_time_energy(frame)
        zcr = compute_zero_crossing_rate(frame)

        energy_list.append(energy)
        zcr_list.append(zcr)

    energy_list = np.array(energy_list)
    zcr_list = np.array(zcr_list)

    # 改进的门限计算方法
    # 估计背景噪声能量：使用前5帧和后5帧的平均能量
    noise_frames = min(5, n_frames // 10)  # 取前后各5帧或总帧数的10%
    if noise_frames > 0:
        noise_energy = np.mean(np.concatenate([
            energy_list[:noise_frames],
            energy_list[-noise_frames:]
        ]))
    else:
        noise_energy = np.min(energy_list)

    # 估计语音能量轮廓：使用能量的高分位数（排除极值）
    speech_energy = np.percentile(energy_list, 90)  # 使用90分位数而非最大值

    # 步骤2：根据语音能量轮廓选取较高的门限T1
    # T1应该使得语音信号的能量包络大部分都在此门限之上
    energy_high_threshold = speech_energy * energy_high_ratio

    # 步骤2续：初判 - 找到能量大于T1的所有帧
    high_energy_frames = np.where(energy_list > energy_high_threshold)[0]

    if len(high_energy_frames) == 0:
        # 如果没有找到高能量帧，返回整个音频
        return 0, len(audio_data), energy_list, zcr_list

    # 初判的起止点位于门限与能量包络交点N3和N4
    N3 = high_energy_frames[0]   # 初判起始点
    N4 = high_energy_frames[-1]  # 初判终止点

    # 步骤3：根据背景噪声能量确定较低的门限T2
    # T2基于背景噪声，而不是最大能量
    energy_low_threshold = noise_energy + (speech_energy - noise_energy) * energy_low_ratio

    # 从初判起点N3往左搜索，找到第一次与门限T2相交的点N2
    N2 = N3
    for i in range(N3 - 1, -1, -1):
        if energy_list[i] <= energy_low_threshold:
            N2 = i + 1  # 相交点的下一帧
            break
    else:
        N2 = 0  # 搜索到头还没找到，说明从开头就是语音

    # 从初判终点N4往右搜索，找到第一次与门限T2相交的点N5
    N5 = N4
    for i in range(N4 + 1, n_frames):
        if energy_list[i] <= energy_low_threshold:
            N5 = i - 1  # 相交点的前一帧
            break
    else:
        N5 = n_frames - 1  # 搜索到尾还没找到，说明到结尾都是语音

    # 步骤4：以短时平均过零率为准，进一步修正起止点
    # 计算过零率门限（基于背景噪声段的过零率）
    if noise_frames > 0:
        noise_zcr = np.mean(np.concatenate([
            zcr_list[:noise_frames],
            zcr_list[-noise_frames:]
        ]))
    else:
        noise_zcr = np.min(zcr_list)

    zcr_threshold = noise_zcr * zcr_threshold_ratio

    # 从N2往左搜索，找到过零率低于阈值T3的点N1
    N1 = N2
    for i in range(N2 - 1, -1, -1):
        if zcr_list[i] <= zcr_threshold:
            N1 = i + 1
            break
    else:
        N1 = 0

    # 从N5往右搜索，找到过零率低于阈值T3的点N6
    N6 = N5
    for i in range(N5 + 1, n_frames):
        if zcr_list[i] <= zcr_threshold:
            N6 = i - 1
            break
    else:
        N6 = n_frames - 1

    # N1和N6就是最终的语音起止点
    start_frame = N1
    end_frame = N6

    # 转换为采样点索引
    start_point = start_frame * frame_shift
    end_point = min(end_frame * frame_shift + frame_length, len(audio_data))

    return start_point, end_point, energy_list, zcr_list


def create_window(window_type, length):
    """
    创建窗函数

    Args:
        window_type: 窗函数类型 ('rectangular', 'hamming', 'hanning')
        length: 窗长度

    Returns:
        窗函数数组
    """
    if window_type == 'rectangular':
        return np.ones(length)
    elif window_type == 'hamming':
        return np.hamming(length)
    elif window_type == 'hanning':
        return np.hanning(length)
    else:
        raise ValueError(f"不支持的窗函数类型: {window_type}")


def frame_signal(audio_data, frame_length, frame_shift, window_type='hamming'):
    """
    分帧加窗

    Args:
        audio_data: 音频数据
        frame_length: 帧长（采样点数）
        frame_shift: 帧移（采样点数）
        window_type: 窗函数类型

    Returns:
        frames: 分帧后的数据，形状为 (n_frames, frame_length)
    """
    n_samples = len(audio_data)

    if n_samples == 0:
        return np.zeros((0, frame_length))

    # 创建窗函数
    window = create_window(window_type, frame_length)

    frames = []
    start = 0
    while start < n_samples:
        end = start + frame_length
        frame = audio_data[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant')
        frames.append(frame * window)

        if end >= n_samples:
            break
        start += frame_shift

    return np.array(frames)


def process_audio_file(filepath, frame_length, frame_shift,
                       window_type='hamming',
                       do_endpoint_detection=True,
                       energy_high_ratio=0.5,
                       energy_low_ratio=0.1,
                       zcr_threshold_ratio=1.5):
    """
    完整的音频处理流程

    Args:
        filepath: WAV文件路径
        frame_length: 帧长
        frame_shift: 帧移
        window_type: 窗函数类型
        do_endpoint_detection: 是否进行端点检测
        energy_high_ratio: 能量高门限比例
        energy_low_ratio: 能量低门限比例
        zcr_threshold_ratio: 过零率门限比例

    Returns:
        frames: 分帧后的数据
        sample_rate: 采样率
        metadata: 元数据字典（包含端点检测信息等）
    """
    # 1. 加载音频
    audio_data, sample_rate = load_wav(filepath)

    # 2. 预处理
    audio_data = preprocess(audio_data)

    metadata = {
        'original_length': len(audio_data),
        'sample_rate': sample_rate
    }

    # 3. 端点检测
    if do_endpoint_detection:
        start_point, end_point, energy_list, zcr_list = endpoint_detection(
            audio_data, frame_length, frame_shift,
            energy_high_ratio, energy_low_ratio, zcr_threshold_ratio
        )

        audio_data = audio_data[start_point:end_point]

        metadata.update({
            'start_point': start_point,
            'end_point': end_point,
            'energy_list': energy_list,
            'zcr_list': zcr_list,
            'segmented_length': len(audio_data)
        })

    if len(audio_data) == 0:
        raise ValueError("No audio remaining after preprocessing and endpoint detection.")

    # 4. 分帧加窗
    frames = frame_signal(audio_data, frame_length, frame_shift, window_type)

    metadata['n_frames'] = len(frames)

    return frames, sample_rate, metadata
