#!/usr/bin/env python3
"""
核心训练模块：简化的训练和评估流程
用于快速训练模型并评估性能
"""
import os
import sys
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import config
from src.audio_processing import process_audio_file
from src.feature_extraction import extract_features_from_frames, normalize_features
from src.models import create_classifier


def load_dataset(data_dir, frame_length_ms=None, frame_shift_ms=None,
                 window_type='hamming', verbose=True):
    """
    加载数据集并提取特征

    Args:
        data_dir: 数据目录路径
        frame_length_ms: 帧长（毫秒），None则使用config中的值
        frame_shift_ms: 帧移（毫秒），None则使用config中的值
        window_type: 窗函数类型
        verbose: 是否打印详细信息

    Returns:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签数组 (n_samples,)
        class_names: 类别名称列表
        feature_names: 特征名称列表
    """
    # 计算帧长和帧移（采样点数）
    if frame_length_ms is None:
        frame_length_ms = config.FRAME_LENGTH_MS
    if frame_shift_ms is None:
        frame_shift_ms = config.FRAME_SHIFT_MS

    frame_length = int(config.SAMPLE_RATE * frame_length_ms / 1000)
    frame_shift = int(config.SAMPLE_RATE * frame_shift_ms / 1000)

    if verbose:
        print(f"\n{'='*60}")
        print(f"加载数据集: {os.path.basename(data_dir)}")
        print(f"帧长: {frame_length_ms}ms ({frame_length} samples)")
        print(f"帧移: {frame_shift_ms}ms ({frame_shift} samples)")
        print(f"窗函数: {window_type}")
        print(f"{'='*60}\n")

    # 获取所有类别文件夹
    class_folders = sorted([d for d in os.listdir(data_dir)
                           if os.path.isdir(os.path.join(data_dir, d))
                           and not d.startswith('.')])

    if verbose:
        print(f"找到 {len(class_folders)} 个类别: {class_folders}\n")

    all_features = []
    all_labels = []

    # 遍历每个类别
    for class_idx, class_name in enumerate(tqdm(class_folders, desc="处理类别", disable=not verbose)):
        class_path = os.path.join(data_dir, class_name)
        wav_files = glob(os.path.join(class_path, '*.wav'))

        # 处理每个音频文件
        for wav_file in wav_files:
            try:
                # 处理音频文件
                frames, _, _ = process_audio_file(
                    wav_file,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    window_type=window_type,
                    do_endpoint_detection=True,
                    energy_high_ratio=config.ENERGY_HIGH_RATIO,
                    energy_low_ratio=config.ENERGY_LOW_RATIO,
                    zcr_threshold_ratio=config.ZCR_THRESHOLD_RATIO
                )

                # 提取特征
                feature_vector, feature_names = extract_features_from_frames(
                    frames, method='statistical'
                )

                all_features.append(feature_vector)
                all_labels.append(class_idx)

            except Exception as e:
                if verbose:
                    print(f"处理失败 {os.path.basename(wav_file)}: {e}")
                continue

    # 转换为numpy数组
    X = np.array(all_features)
    y = np.array(all_labels)

    if verbose:
        print(f"\n数据集加载成功！")
        print(f"总样本数: {len(X)}")
        print(f"特征维度: {X.shape[1]}")
        print(f"类别分布: {dict(zip(class_folders, np.bincount(y)))}\n")

    return X, y, class_folders, feature_names


def train_and_evaluate(X, y, classifier_type='mlp', test_size=0.2,
                      random_seed=42, verbose=True, **classifier_params):
    """
    训练并评估模型

    Args:
        X: 特征矩阵
        y: 标签数组
        classifier_type: 分类器类型 ('knn', 'svm', 'mlp', 'naive_bayes', 'decision_tree')
        test_size: 测试集比例
        random_seed: 随机种子
        verbose: 是否打印详细信息
        **classifier_params: 分类器参数

    Returns:
        results: 包含以下字段的字典
            - accuracy: 测试集准确率
            - train_accuracy: 训练集准确率（如果适用）
            - predictions: 预测标签
            - confusion_matrix: 混淆矩阵
            - classification_report: 分类报告
            - classifier: 训练好的分类器对象
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"训练分类器: {classifier_type.upper()}")
        print(f"{'='*60}\n")

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    # 特征归一化
    X_train_norm, mean, std = normalize_features(X_train)
    X_test_norm, _, _ = normalize_features(X_test, mean, std)

    if verbose:
        print(f"训练集: {len(X_train)} 样本")
        print(f"测试集: {len(X_test)} 样本\n")

    # 创建分类器
    if classifier_type == 'mlp':
        # MLP特殊处理
        num_classes = len(np.unique(y))

        # 从classifier_params获取参数，否则使用config默认值
        learning_rate = classifier_params.get('learning_rate', config.MLP_LEARNING_RATE)
        hidden_layers = classifier_params.get('hidden_layers', config.MLP_HIDDEN_LAYERS)
        epochs = classifier_params.get('epochs', config.MLP_EPOCHS)
        batch_size = classifier_params.get('batch_size', config.MLP_BATCH_SIZE)

        classifier = create_classifier(
            'mlp',
            input_size=X_train.shape[1],
            hidden_layers=hidden_layers,
            num_classes=num_classes,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size
        )

        if verbose:
            print(f"MLP参数:")
            print(f"  学习率: {learning_rate}")
            print(f"  隐藏层: {hidden_layers}")
            print(f"  训练轮数: {epochs}")
            print(f"  批大小: {batch_size}\n")

        classifier.fit(X_train_norm, y_train, verbose=verbose)
    else:
        # 其他传统分类器
        classifier = create_classifier(classifier_type, **classifier_params)
        classifier.fit(X_train_norm, y_train)

    # 评估
    results = classifier.evaluate(X_test_norm, y_test)
    results['classifier'] = classifier

    # 计算训练集准确率
    if classifier_type == 'mlp':
        train_pred = classifier.predict(X_train_norm)
    else:
        train_pred = classifier.predict(X_train_norm)

    from sklearn.metrics import accuracy_score
    results['train_accuracy'] = accuracy_score(y_train, train_pred)

    if verbose:
        print(f"\n结果:")
        print(f"  训练集准确率: {results['train_accuracy']:.4f}")
        print(f"  测试集准确率: {results['accuracy']:.4f}")
        print(f"{'='*60}\n")

    return results


def quick_experiment(data_dir=None, classifier_type='mlp', frame_length_ms=None,
                    frame_shift_ms=None, window_type='hamming', verbose=True,
                    **classifier_params):
    """
    快速实验：一次性完成数据加载、训练和评估

    Args:
        data_dir: 数据目录，None则使用config中的DATA_DIR
        classifier_type: 分类器类型
        frame_length_ms: 帧长（毫秒）
        frame_shift_ms: 帧移（毫秒）
        window_type: 窗函数类型
        verbose: 是否打印详细信息
        **classifier_params: 分类器参数

    Returns:
        results: 实验结果字典
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    # 加载数据
    X, y, class_names, feature_names = load_dataset(
        data_dir=data_dir,
        frame_length_ms=frame_length_ms,
        frame_shift_ms=frame_shift_ms,
        window_type=window_type,
        verbose=verbose
    )

    # 训练和评估
    results = train_and_evaluate(
        X, y,
        classifier_type=classifier_type,
        test_size=config.TEST_SIZE,
        random_seed=config.RANDOM_SEED,
        verbose=verbose,
        **classifier_params
    )

    # 添加额外信息
    results['class_names'] = class_names
    results['feature_names'] = feature_names
    results['dataset'] = os.path.basename(data_dir)
    results['frame_length_ms'] = frame_length_ms or config.FRAME_LENGTH_MS
    results['frame_shift_ms'] = frame_shift_ms or config.FRAME_SHIFT_MS
    results['window_type'] = window_type

    return results


if __name__ == '__main__':
    """
    示例用法：直接运行此脚本进行快速实验
    """
    print("="*60)
    print("快速实验示例")
    print("="*60)

    # 示例1：使用默认参数
    print("\n示例1: 使用默认参数训练MLP")
    results = quick_experiment(classifier_type='mlp')
    print(f"准确率: {results['accuracy']:.4f}")

    # 示例2：修改学习率
    print("\n示例2: 使用不同学习率训练MLP")
    results = quick_experiment(classifier_type='mlp', learning_rate=0.01)
    print(f"准确率: {results['accuracy']:.4f}")

    # 示例3：使用SVM
    print("\n示例3: 使用SVM")
    results = quick_experiment(classifier_type='svm')
    print(f"准确率: {results['accuracy']:.4f}")
