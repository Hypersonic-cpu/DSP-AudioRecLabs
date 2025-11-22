#!/usr/bin/env python3
"""
消融实验：对比不同超参数对模型性能的影响
- 学习率对比
- 帧长对比
- 帧移对比
"""
import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from datetime import datetime

import config
from train_model import load_dataset, train_and_evaluate

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def ablation_learning_rate(data_dir=None, learning_rates=None, classifier_type='mlp',
                           save_dir=None, verbose=True):
    """
    消融实验：学习率对比

    Args:
        data_dir: 数据目录
        learning_rates: 学习率列表
        classifier_type: 分类器类型（推荐使用mlp）
        save_dir: 结果保存目录
        verbose: 是否打印详细信息

    Returns:
        results: 实验结果字典
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    if learning_rates is None:
        learning_rates = config.LEARNING_RATES
    if save_dir is None:
        save_dir = os.path.join(config.RESULTS_DIR, 'ablation_learning_rate')

    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*70)
    print("消融实验：学习率对比")
    print("="*70)
    print(f"数据集: {os.path.basename(data_dir)}")
    print(f"学习率范围: {learning_rates}")
    print(f"分类器: {classifier_type}")
    print("="*70 + "\n")

    # 只加载一次数据
    X, y, class_names, feature_names = load_dataset(
        data_dir=data_dir,
        window_type='hamming',
        verbose=verbose
    )

    results = {}

    for lr in learning_rates:
        print(f"\n{'='*60}")
        print(f"测试学习率: {lr}")
        print(f"{'='*60}")

        result = train_and_evaluate(
            X, y,
            classifier_type=classifier_type,
            learning_rate=lr,
            verbose=verbose
        )

        results[lr] = {
            'accuracy': result['accuracy'],
            'train_accuracy': result['train_accuracy'],
            'confusion_matrix': result['confusion_matrix'].tolist(),
        }

        print(f"准确率: {result['accuracy']:.4f}\n")

    # 可视化结果
    _plot_ablation_results(
        results,
        x_label='Learning Rate',
        title=f'Learning Rate Ablation Study - {os.path.basename(data_dir)}',
        save_path=os.path.join(save_dir, 'learning_rate_comparison.png'),
        x_scale='log'
    )

    # 保存详细结果
    _save_ablation_results(
        results,
        save_dir=save_dir,
        experiment_name='学习率对比',
        param_name='learning_rate',
        dataset=os.path.basename(data_dir),
        class_names=class_names
    )

    print(f"\n结果已保存到: {save_dir}\n")

    return results


def ablation_frame_length(data_dir=None, frame_lengths_ms=None, classifier_type='mlp',
                         save_dir=None, verbose=True):
    """
    消融实验：帧长对比

    Args:
        data_dir: 数据目录
        frame_lengths_ms: 帧长列表（毫秒）
        classifier_type: 分类器类型
        save_dir: 结果保存目录
        verbose: 是否打印详细信息

    Returns:
        results: 实验结果字典
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    if frame_lengths_ms is None:
        frame_lengths_ms = config.FRAME_LENGTH_MS_RANGE
    if save_dir is None:
        save_dir = os.path.join(config.RESULTS_DIR, 'ablation_frame_length')

    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*70)
    print("消融实验：帧长对比")
    print("="*70)
    print(f"数据集: {os.path.basename(data_dir)}")
    print(f"帧长范围: {frame_lengths_ms} ms")
    print(f"分类器: {classifier_type}")
    print("="*70 + "\n")

    results = {}

    for frame_len_ms in frame_lengths_ms:
        print(f"\n{'='*60}")
        print(f"测试帧长: {frame_len_ms} ms")
        print(f"{'='*60}")

        # 重新加载数据（因为帧长会影响特征提取）
        X, y, class_names, feature_names = load_dataset(
            data_dir=data_dir,
            frame_length_ms=frame_len_ms,
            window_type='hamming',
            verbose=verbose
        )

        result = train_and_evaluate(
            X, y,
            classifier_type=classifier_type,
            verbose=verbose
        )

        results[frame_len_ms] = {
            'accuracy': result['accuracy'],
            'train_accuracy': result['train_accuracy'],
            'confusion_matrix': result['confusion_matrix'].tolist(),
        }

        print(f"准确率: {result['accuracy']:.4f}\n")

    # 可视化结果
    _plot_ablation_results(
        results,
        x_label='Frame Length (ms)',
        title=f'Frame Length Ablation Study - {os.path.basename(data_dir)}',
        save_path=os.path.join(save_dir, 'frame_length_comparison.png')
    )

    # 保存详细结果
    _save_ablation_results(
        results,
        save_dir=save_dir,
        experiment_name='帧长对比',
        param_name='frame_length_ms',
        dataset=os.path.basename(data_dir),
        class_names=class_names
    )

    print(f"\n结果已保存到: {save_dir}\n")

    return results


def ablation_frame_shift(data_dir=None, frame_shifts_ms=None, classifier_type='mlp',
                        save_dir=None, verbose=True):
    """
    消融实验：帧移对比

    Args:
        data_dir: 数据目录
        frame_shifts_ms: 帧移列表（毫秒）
        classifier_type: 分类器类型
        save_dir: 结果保存目录
        verbose: 是否打印详细信息

    Returns:
        results: 实验结果字典
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    if frame_shifts_ms is None:
        frame_shifts_ms = config.FRAME_SHIFT_MS_RANGE
    if save_dir is None:
        save_dir = os.path.join(config.RESULTS_DIR, 'ablation_frame_shift')

    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*70)
    print("消融实验：帧移对比")
    print("="*70)
    print(f"数据集: {os.path.basename(data_dir)}")
    print(f"帧移范围: {frame_shifts_ms} ms")
    print(f"分类器: {classifier_type}")
    print("="*70 + "\n")

    results = {}

    for frame_shift_ms in frame_shifts_ms:
        print(f"\n{'='*60}")
        print(f"测试帧移: {frame_shift_ms} ms")
        print(f"{'='*60}")

        # 重新加载数据（因为帧移会影响特征提取）
        X, y, class_names, feature_names = load_dataset(
            data_dir=data_dir,
            frame_shift_ms=frame_shift_ms,
            window_type='hamming',
            verbose=verbose
        )

        result = train_and_evaluate(
            X, y,
            classifier_type=classifier_type,
            verbose=verbose
        )

        results[frame_shift_ms] = {
            'accuracy': result['accuracy'],
            'train_accuracy': result['train_accuracy'],
            'confusion_matrix': result['confusion_matrix'].tolist(),
        }

        print(f"准确率: {result['accuracy']:.4f}\n")

    # 可视化结果
    _plot_ablation_results(
        results,
        x_label='Frame Shift (ms)',
        title=f'Frame Shift Ablation Study - {os.path.basename(data_dir)}',
        save_path=os.path.join(save_dir, 'frame_shift_comparison.png')
    )

    # 保存详细结果
    _save_ablation_results(
        results,
        save_dir=save_dir,
        experiment_name='帧移对比',
        param_name='frame_shift_ms',
        dataset=os.path.basename(data_dir),
        class_names=class_names
    )

    print(f"\n结果已保存到: {save_dir}\n")

    return results


def _plot_ablation_results(results, x_label, title, save_path, x_scale='linear'):
    """
    绘制消融实验结果（美化版，使用英文）

    Args:
        results: 结果字典 {param_value: {'accuracy': ..., 'train_accuracy': ...}}
        x_label: x轴标签
        title: 图表标题
        save_path: 保存路径
        x_scale: x轴刻度类型 ('linear' 或 'log')
    """
    params = list(results.keys())
    test_accs = [results[p]['accuracy'] for p in params]
    train_accs = [results[p]['train_accuracy'] for p in params]

    # 使用更大的图表，更好的配色
    fig, ax = plt.subplots(figsize=(14, 7))

    # 设置漂亮的配色方案
    color_train = '#2E86AB'  # 深蓝色
    color_test = '#A23B72'   # 紫红色
    color_best = '#F18F01'   # 橙色

    # 绘制训练集和测试集准确率，使用更粗的线条和更大的标记
    ax.plot(params, train_accs, marker='o', linewidth=2.5, markersize=8,
            label='Training Accuracy', color=color_train, alpha=0.8)
    ax.plot(params, test_accs, marker='s', linewidth=2.5, markersize=8,
            label='Test Accuracy', color=color_test, alpha=0.8)

    # 标注最佳值
    best_idx = np.argmax(test_accs)
    best_param = params[best_idx]
    best_acc = test_accs[best_idx]

    # 绘制最佳值的垂直线和高亮点
    ax.axvline(best_param, color=color_best, linestyle='--', alpha=0.6,
               linewidth=2, label=f'Best Value: {best_param}')
    ax.scatter([best_param], [best_acc], color=color_best, s=400,
               zorder=5, marker='*', edgecolors='white', linewidths=2)

    # 在最佳点添加文本框
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor=color_best,
                      alpha=0.8, edgecolor='white', linewidth=2)
    ax.text(best_param, best_acc + 0.02, f'Best: {best_acc:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color='white', bbox=bbox_props)

    # 设置标签和标题（英文）
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)

    # 设置x轴刻度
    ax.set_xscale(x_scale)

    # 美化图例
    legend = ax.legend(fontsize=12, loc='best', framealpha=0.95,
                       edgecolor='gray', fancybox=True, shadow=True)

    # 设置y轴范围，留出空间显示标注
    y_min = min(min(train_accs), min(test_accs))
    y_max = max(max(train_accs), max(test_accs))
    y_range = y_max - y_min
    ax.set_ylim([max(0, y_min - 0.05 * y_range), min(1, y_max + 0.1 * y_range)])

    # 设置刻度字体
    ax.tick_params(axis='both', labelsize=11)

    # 设置背景色
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Plot saved: {save_path}")


def _save_ablation_results(results, save_dir, experiment_name, param_name, dataset, class_names):
    """
    保存消融实验详细结果

    Args:
        results: 结果字典
        save_dir: 保存目录
        experiment_name: 实验名称
        param_name: 参数名称
        dataset: 数据集名称
        class_names: 类别名称列表
    """
    # 保存JSON格式
    json_path = os.path.join(save_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': experiment_name,
            'dataset': dataset,
            'param_name': param_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': {str(k): v for k, v in results.items()}
        }, f, indent=2, ensure_ascii=False)

    # 保存文本格式
    txt_path = os.path.join(save_dir, 'results_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"{experiment_name}\n")
        f.write("="*70 + "\n\n")
        f.write(f"数据集: {dataset}\n")
        f.write(f"类别: {', '.join(class_names)}\n")
        f.write(f"参数: {param_name}\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"{'参数值':<15} {'训练准确率':<15} {'测试准确率':<15}\n")
        f.write("-"*70 + "\n")

        for param, result in results.items():
            f.write(f"{str(param):<15} {result['train_accuracy']:<15.4f} {result['accuracy']:<15.4f}\n")

        # 找出最佳参数
        best_param = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_acc = results[best_param]['accuracy']

        f.write("\n" + "="*70 + "\n")
        f.write(f"最佳参数: {best_param}\n")
        f.write(f"最佳准确率: {best_acc:.4f}\n")
        f.write("="*70 + "\n")

    print(f"结果已保存:")
    print(f"  - JSON: {json_path}")
    print(f"  - TXT:  {txt_path}")


def main():
    """主函数：运行消融实验"""
    parser = argparse.ArgumentParser(description='消融实验：对比不同超参数')

    parser.add_argument(
        '--experiment',
        type=str,
        default='all',
        choices=['all', 'lr', 'frame_length', 'frame_shift'],
        help='实验类型：all（全部）, lr（学习率）, frame_length（帧长）, frame_shift（帧移）'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['name', 'number'],
        help='数据集类型：name（中文名字）, number（数字）'
    )

    parser.add_argument(
        '--classifier',
        type=str,
        default='mlp',
        choices=['mlp', 'knn', 'svm', 'decision_tree', 'naive_bayes'],
        help='分类器类型（默认：mlp）'
    )

    args = parser.parse_args()

    # 确定数据集路径
    if args.dataset:
        data_dir = config.DATASET_PATHS[args.dataset]
    else:
        data_dir = config.DATA_DIR

    print("\n" + "="*70)
    print("DSP消融实验")
    print("="*70)
    print(f"数据集: {os.path.basename(data_dir)}")
    print(f"分类器: {args.classifier}")
    print("="*70 + "\n")

    # 运行实验
    if args.experiment == 'all':
        print("运行所有消融实验...\n")

        # 1. 学习率对比
        if args.classifier == 'mlp':
            ablation_learning_rate(data_dir=data_dir, classifier_type=args.classifier)
        else:
            print("注意: 学习率实验仅适用于MLP，跳过...\n")

        # 2. 帧长对比
        ablation_frame_length(data_dir=data_dir, classifier_type=args.classifier)

        # 3. 帧移对比
        ablation_frame_shift(data_dir=data_dir, classifier_type=args.classifier)

    elif args.experiment == 'lr':
        if args.classifier != 'mlp':
            print("警告: 学习率实验仅适用于MLP")
            return

        ablation_learning_rate(data_dir=data_dir, classifier_type=args.classifier)

    elif args.experiment == 'frame_length':
        ablation_frame_length(data_dir=data_dir, classifier_type=args.classifier)

    elif args.experiment == 'frame_shift':
        ablation_frame_shift(data_dir=data_dir, classifier_type=args.classifier)

    print("\n" + "="*70)
    print("消融实验完成！")
    print(f"结果保存在: {config.RESULTS_DIR}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
