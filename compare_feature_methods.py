#!/usr/bin/env python3
"""
特征提取方法对比实验：
1. 统计特征法（降维到15维）vs 序列特征法（完整序列 + padding）
2. 只使用能量和过零率两个指标
"""
import os
import sys
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.time_domain.audio_processing import process_audio_file
from src.time_domain.feature_extraction import extract_features_from_frames, normalize_features, pad_or_truncate_sequence
from src.time_domain.models import create_classifier

print("="*70)
print("特征提取方法对比实验")
print("="*70)
print()

# ==================== 加载数据 ====================
print("Step 1: 加载数据集...")

class_folders = sorted([d for d in os.listdir(config.DATA_DIR)
                       if os.path.isdir(os.path.join(config.DATA_DIR, d))
                       and not d.startswith('.')])

print(f"找到 {len(class_folders)} 个类别\n")

# 方法1：统计特征（降维）
print("方法1: 统计特征法（降维到固定维度）")
print("-"*70)

X_statistical = []
y_labels = []

for class_idx, class_name in enumerate(tqdm(class_folders, desc="提取统计特征")):
    class_path = os.path.join(config.DATA_DIR, class_name)
    wav_files = glob(os.path.join(class_path, '*.wav'))

    for wav_file in wav_files:
        try:
            frames, _, _ = process_audio_file(
                wav_file,
                config.FRAME_LENGTH,
                config.FRAME_SHIFT,
                window_type='hamming',
                do_endpoint_detection=True
            )

            # 提取统计特征
            feature_vector, _ = extract_features_from_frames(frames, method='statistical')
            X_statistical.append(feature_vector)
            y_labels.append(class_idx)

        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue

X_statistical = np.array(X_statistical)
y_labels = np.array(y_labels)

print(f"✓ 统计特征形状: {X_statistical.shape}")
print(f"  每个样本: {X_statistical.shape[1]} 维固定特征\n")

# 方法2：序列特征（只用能量和过零率）
print("方法2: 序列特征法（完整序列，只用能量+过零率）")
print("-"*70)

X_sequences = []
sequence_lengths = []

for class_idx, class_name in enumerate(tqdm(class_folders, desc="提取序列特征")):
    class_path = os.path.join(config.DATA_DIR, class_name)
    wav_files = glob(os.path.join(class_path, '*.wav'))

    for wav_file in wav_files:
        try:
            frames, _, _ = process_audio_file(
                wav_file,
                config.FRAME_LENGTH,
                config.FRAME_SHIFT,
                window_type='hamming',
                do_endpoint_detection=True
            )

            # 提取序列特征（只用能量和过零率）
            feature_seq, _ = extract_features_from_frames(
                frames,
                method='sequence',
                use_only_energy_zcr=True  # 只用能量和过零率
            )

            X_sequences.append(feature_seq)
            sequence_lengths.append(len(feature_seq))

        except Exception as e:
            continue

# 对齐序列长度（padding/truncating）
print(sequence_lengths)
max_length = max(sequence_lengths)
print(f"✓ 序列长度范围: {min(sequence_lengths)} ~ {max_length} 帧")
print(f"  使用padding对齐到: {max_length} 帧")

X_sequences_padded = np.array([
    pad_or_truncate_sequence(seq, max_length)
    for seq in X_sequences
])

print(f"✓ 序列特征形状: {X_sequences_padded.shape}")
print(f"  每个样本: {max_length} 帧 × 2 特征（能量+过零率）")
print(f"  展平后: {max_length * 2} 维\n")

# 展平序列特征用于传统分类器
X_sequences_flat = X_sequences_padded.reshape(len(X_sequences_padded), -1)

# ==================== 训练与评估 ====================
print("="*70)
print("训练分类器并对比")
print("="*70)
print()

# 划分训练/测试集
X_stat_train, X_stat_test, y_train, y_test = train_test_split(
    X_statistical, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

X_seq_train, X_seq_test, _, _ = train_test_split(
    X_sequences_flat, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

# 特征归一化
X_stat_train, mean_stat, std_stat = normalize_features(X_stat_train)
X_stat_test, _, _ = normalize_features(X_stat_test, mean_stat, std_stat)

X_seq_train, mean_seq, std_seq = normalize_features(X_seq_train)
X_seq_test, _, _ = normalize_features(X_seq_test, mean_seq, std_seq)

print(f"训练集: {len(X_stat_train)} 样本")
print(f"测试集: {len(X_stat_test)} 样本\n")

# 定义要测试的分类器
classifiers = {
    'KNN': ('knn', {'n_neighbors': 3}),
    'SVM': ('svm', {'C': 1.0, 'kernel': 'rbf'}),
    'Decision Tree': ('decision_tree', {}), 
}

results = {}

for clf_name, (clf_type, params) in classifiers.items():
    print(f"\n{'='*70}")
    print(f"分类器: {clf_name}")
    print(f"{'='*70}")

    # 方法1：统计特征
    print(f"\n  [方法1] 统计特征 (15维固定特征)")
    clf_stat = create_classifier(clf_type, **params)
    clf_stat.fit(X_stat_train, y_train)
    result_stat = clf_stat.evaluate(X_stat_test, y_test)
    acc_stat = result_stat['accuracy']
    print(f"    准确率: {acc_stat:.4f}")

    # 方法2：序列特征
    print(f"\n  [方法2] 序列特征 ({max_length}帧 × 2特征 = {max_length*2}维)")
    clf_seq = create_classifier(clf_type, **params)
    clf_seq.fit(X_seq_train, y_train)
    result_seq = clf_seq.evaluate(X_seq_test, y_test)
    acc_seq = result_seq['accuracy']
    print(f"    准确率: {acc_seq:.4f}")

    # 对比
    diff = acc_seq - acc_stat
    print(f"\n  对比: 序列法 vs 统计法 = {diff:+.4f}")

    results[clf_name] = {
        'statistical': acc_stat,
        'sequence': acc_seq,
        'diff': diff
    }

# ==================== 结果总结 ====================
print("\n" + "="*70)
print("实验总结")
print("="*70)
print()

print(f"{'分类器':<20} {'统计特征':<15} {'序列特征':<15} {'差异':<10}")
print("-"*70)
for clf_name, result in results.items():
    print(f"{clf_name:<20} {result['statistical']:<15.4f} {result['sequence']:<15.4f} {result['diff']:<+10.4f}")

print("\n特征对比：")
print(f"  统计特征: {X_statistical.shape[1]} 维（3种特征 × 5统计量）")
print(f"  序列特征: {max_length * 2} 维（{max_length}帧 × 2特征：能量+过零率）")

print("\n结论：")
avg_stat = np.mean([r['statistical'] for r in results.values()])
avg_seq = np.mean([r['sequence'] for r in results.values()])
print(f"  平均准确率 - 统计特征: {avg_stat:.4f}")
print(f"  平均准确率 - 序列特征: {avg_seq:.4f}")
if avg_seq > avg_stat:
    print(f"  → 序列特征法更优，提升 {avg_seq - avg_stat:.4f}")
else:
    print(f"  → 统计特征法更优，提升 {avg_stat - avg_seq:.4f}")

print("\n" + "="*70)
