"""
特征提取模块：提取时域特征（能量、幅度、过零率）
"""
import numpy as np
from src.time_domain.audio_processing import (
    compute_short_time_energy,
    compute_short_time_magnitude,
    compute_zero_crossing_rate
)


def extract_frame_features(frames):
    """
    为每一帧提取时域特征

    Args:
        frames: 分帧后的音频数据，形状为 (n_frames, frame_length)

    Returns:
        features: 特征字典
            - 'energy': 短时能量序列
            - 'magnitude': 短时平均幅度序列
            - 'zcr': 短时过零率序列
    """
    n_frames = len(frames)

    if n_frames == 0:
        raise ValueError("No frames provided for feature extraction.")

    energy_seq = np.zeros(n_frames)
    magnitude_seq = np.zeros(n_frames)
    zcr_seq = np.zeros(n_frames)

    for i, frame in enumerate(frames):
        energy_seq[i] = compute_short_time_energy(frame)
        magnitude_seq[i] = compute_short_time_magnitude(frame)
        zcr_seq[i] = compute_zero_crossing_rate(frame)

    return {
        'energy': energy_seq,
        'magnitude': magnitude_seq,
        'zcr': zcr_seq
    }


def compute_statistics(sequence):
    """
    计算序列的统计特征

    Args:
        sequence: 特征序列

    Returns:
        stats: 统计特征字典
    """
    return {
        'mean': np.mean(sequence),
        'std': np.std(sequence),
        'max': np.max(sequence),
        'min': np.min(sequence),
        'median': np.median(sequence)
    }


def extract_statistical_features(frame_features):
    """
    从帧级特征序列中提取统计特征

    Args:
        frame_features: 帧级特征字典（来自extract_frame_features）

    Returns:
        feature_vector: 统计特征向量（一维数组）
        feature_names: 特征名称列表
    """
    feature_vector = []
    feature_names = []

    # 对每种特征类型提取统计量
    for feature_type in ['energy', 'magnitude', 'zcr']:
        sequence = frame_features[feature_type]
        stats = compute_statistics(sequence)

        for stat_name, stat_value in stats.items():
            feature_vector.append(stat_value)
            feature_names.append(f'{feature_type}_{stat_name}')

    return np.array(feature_vector), feature_names


def extract_features_from_frames(frames, method='statistical', use_only_energy_zcr=False):
    """
    从分帧后的音频提取特征

    Args:
        frames: 分帧后的音频数据
        method: 特征提取方法
            - 'statistical': 提取统计特征（默认，降维到固定维度）
            - 'sequence': 返回完整序列（需要后续对齐到相同长度）
        use_only_energy_zcr: 是否只使用能量和过零率（忽略幅度）

    Returns:
        features: 特征向量或特征序列
        feature_names: 特征名称列表（仅statistical方法）
    """
    # 提取帧级特征
    frame_features = extract_frame_features(frames)

    if method == 'statistical':
        # 提取统计特征（降维）
        feature_vector, feature_names = extract_statistical_features(frame_features)
        return feature_vector, feature_names

    elif method == 'sequence':
        # 返回完整序列（不降维）
        if use_only_energy_zcr:
            # 只使用能量和过零率（按要求）
            feature_seq = np.stack([
                frame_features['energy'],
                frame_features['zcr']
            ], axis=1)  # 形状：(n_frames, 2)
        else:
            # 使用所有三种特征
            feature_seq = np.stack([
                frame_features['energy'],
                frame_features['magnitude'],
                frame_features['zcr']
            ], axis=1)  # 形状：(n_frames, 3)
        return feature_seq, None

    else:
        raise ValueError(f"不支持的特征提取方法: {method}")


def pad_or_truncate_sequence(sequence, target_length):
    """
    填充或截断序列到目标长度

    Args:
        sequence: 输入序列，形状 (n_frames, n_features)
        target_length: 目标帧数

    Returns:
        处理后的序列，形状 (target_length, n_features)
    """
    current_length = len(sequence)

    if current_length < target_length:
        # 填充（用零填充）
        padding = np.zeros((target_length - current_length, sequence.shape[1]))
        return np.vstack([sequence, padding])
    else:
        # 截断
        return sequence[:target_length]


def normalize_features(features, mean=None, std=None):
    """
    特征归一化（Z-score标准化）

    Args:
        features: 特征向量或特征矩阵
        mean: 均值（如果为None，则从features计算）
        std: 标准差（如果为None，则从features计算）

    Returns:
        normalized_features: 归一化后的特征
        mean: 均值
        std: 标准差
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)

    # 防止除零
    std = np.where(std == 0, 1, std)

    normalized_features = (features - mean) / std

    return normalized_features, mean, std
