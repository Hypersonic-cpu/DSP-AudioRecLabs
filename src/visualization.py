"""
可视化模块：生成各种实验结果图表
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import os
import sys

# 设置中文字体支持
def setup_chinese_fonts():
    """配置matplotlib以支持中文显示"""
    # macOS系统优先使用的中文字体列表
    chinese_fonts = [
        'Arial Unicode MS',    # macOS
        'PingFang SC',         # macOS (苹方)
        'Heiti SC',            # macOS (黑体-简)
        'STSong',              # macOS (宋体)
        'SimHei',              # Windows/Linux (黑体)
        'Microsoft YaHei',     # Windows (微软雅黑)
        'WenQuanYi Micro Hei', # Linux (文泉驿微米黑)
        'DejaVu Sans',         # 备用
    ]

    # 设置字体
    rcParams['font.sans-serif'] = chinese_fonts
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 设置默认字体大小
    rcParams['font.size'] = 10
    rcParams['axes.titlesize'] = 12
    rcParams['axes.labelsize'] = 11
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10

# 初始化字体配置
setup_chinese_fonts()

# 设置seaborn风格
sns.set_style("whitegrid")


def plot_waveform(audio_data, sample_rate, title="音频波形", save_path=None):
    """
    绘制音频波形

    Args:
        audio_data: 音频数据
        sample_rate: 采样率
        title: 图表标题
        save_path: 保存路径
    """
    time = np.arange(len(audio_data)) / sample_rate

    plt.figure(figsize=(12, 4))
    plt.plot(time, audio_data, linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_endpoint_detection(audio_data, sample_rate, start_point, end_point,
                           energy_list, zcr_list, frame_shift,
                           title="Endpoint Detection", save_path=None):
    """
    可视化端点检测结果

    Args:
        audio_data: 原始音频数据
        sample_rate: 采样率
        start_point: 检测到的起始点
        end_point: 检测到的终止点
        energy_list: 能量序列
        zcr_list: 过零率序列
        frame_shift: 帧移
        title: 图表标题
        save_path: 保存路径
    """
    time = np.arange(len(audio_data)) / sample_rate
    frame_time = np.arange(len(energy_list)) * frame_shift / sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # 原始波形
    axes[0].plot(time, audio_data, linewidth=0.5, color='blue', alpha=0.7)
    axes[0].axvline(start_point / sample_rate, color='red', linestyle='--', label='Start')
    axes[0].axvline(end_point / sample_rate, color='green', linestyle='--', label='End')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Waveform')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 短时能量
    axes[1].plot(frame_time, energy_list, linewidth=1, color='orange')
    axes[1].axvline(start_point / sample_rate, color='red', linestyle='--')
    axes[1].axvline(end_point / sample_rate, color='green', linestyle='--')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Energy')
    axes[1].set_title('Short-Time Energy')
    axes[1].grid(True, alpha=0.3)

    # 短时过零率
    axes[2].plot(frame_time, zcr_list, linewidth=1, color='purple')
    axes[2].axvline(start_point / sample_rate, color='red', linestyle='--')
    axes[2].axvline(end_point / sample_rate, color='green', linestyle='--')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('ZCR')
    axes[2].set_title('Zero Crossing Rate')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_frame_features(frame_features, title="Frame Features", save_path=None):
    """
    绘制帧级特征

    Args:
        frame_features: 特征字典（包含energy, magnitude, zcr）
        title: 图表标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # 短时能量
    axes[0].plot(frame_features['energy'], linewidth=1, color='orange')
    axes[0].set_ylabel('Energy')
    axes[0].set_title('Short-Time Energy')
    axes[0].grid(True, alpha=0.3)

    # 短时平均幅度
    axes[1].plot(frame_features['magnitude'], linewidth=1, color='blue')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Short-Time Magnitude')
    axes[1].grid(True, alpha=0.3)

    # 短时过零率
    axes[2].plot(frame_features['zcr'], linewidth=1, color='purple')
    axes[2].set_xlabel('Frame Index')
    axes[2].set_ylabel('ZCR')
    axes[2].set_title('Zero Crossing Rate')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """
    绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表（支持中文）
        title: 图表标题
        save_path: 保存路径
    """
    # 确保中文字体设置已应用
    setup_chinese_fonts()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label (预测标签)', fontsize=11)
    plt.ylabel('True Label (真实标签)', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')

    # 旋转x轴标签以更好地显示中文
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_classifier_comparison(results_dict, metric='accuracy',
                              title="Classifier Comparison", save_path=None):
    """
    比较不同分类器的性能

    Args:
        results_dict: 结果字典，格式为 {classifier_name: results}
        metric: 要比较的指标（'accuracy'）
        title: 图表标题
        save_path: 保存路径
    """
    classifier_names = list(results_dict.keys())
    scores = [results_dict[name][metric] for name in classifier_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(classifier_names, scores, color=sns.color_palette("husl", len(classifier_names)))

    # 在柱状图上添加数值
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10)

    plt.xlabel('Classifier')
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.ylim([0, 1.1])
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_window_comparison(results_dict, metric='accuracy',
                          title="Window Function Comparison", save_path=None):
    """
    比较不同窗函数的影响

    Args:
        results_dict: 结果字典，格式为 {window_type: {classifier_name: results}}
        metric: 要比较的指标
        title: 图表标题
        save_path: 保存路径
    """
    window_types = list(results_dict.keys())
    classifier_names = list(results_dict[window_types[0]].keys())

    # 准备数据
    data = np.zeros((len(window_types), len(classifier_names)))
    for i, window_type in enumerate(window_types):
        for j, classifier_name in enumerate(classifier_names):
            data[i, j] = results_dict[window_type][classifier_name][metric]

    # 绘制分组柱状图
    x = np.arange(len(window_types))
    width = 0.8 / len(classifier_names)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, classifier_name in enumerate(classifier_names):
        offset = (i - len(classifier_names) / 2) * width + width / 2
        bars = ax.bar(x + offset, data[:, i], width, label=classifier_name)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Window Type')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(window_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_mlp_training_history(train_losses, train_accuracies,
                              title="MLP Training History", save_path=None):
    """
    绘制MLP训练历史

    Args:
        train_losses: 训练损失列表
        train_accuracies: 训练准确率列表
        title: 图表标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    axes[0].plot(train_losses, linewidth=2, color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(train_accuracies, linewidth=2, color='blue')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_feature_distribution(features, labels, feature_names,
                             title="Feature Distribution", save_path=None,
                             class_names=None):
    """
    绘制特征分布

    Args:
        features: 特征矩阵
        labels: 标签
        feature_names: 特征名称列表
        title: 图表标题
        save_path: 保存路径
        class_names: 类别名称列表（可选，支持中文）
    """
    # 确保中文字体设置已应用
    setup_chinese_fonts()

    n_features = min(len(feature_names), 9)  # 最多显示9个特征

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(n_features):
        for label in np.unique(labels):
            mask = labels == label
            # 使用类别名称（如果提供）或标签索引
            if class_names is not None and label < len(class_names):
                label_text = class_names[label]
            else:
                label_text = f'Class {label}'

            axes[i].hist(features[mask, i], alpha=0.6, label=label_text, bins=20)

        axes[i].set_xlabel(feature_names[i])
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(n_features, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
