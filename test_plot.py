#!/usr/bin/env python3
"""
测试图表生成：验证美化后的图表样式
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# 设置样式
sns.set_style("whitegrid")

def test_plot():
    """测试绘制美化后的消融实验图表"""

    # 模拟数据
    learning_rates = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08]
    train_accs = [0.75, 0.82, 0.86, 0.90, 0.93, 0.95, 0.96, 0.97, 0.96, 0.94, 0.90]
    test_accs = [0.70, 0.78, 0.82, 0.86, 0.89, 0.91, 0.90, 0.88, 0.85, 0.82, 0.78]

    # 使用更大的图表，更好的配色
    fig, ax = plt.subplots(figsize=(14, 7))

    # 设置漂亮的配色方案
    color_train = '#2E86AB'  # 深蓝色
    color_test = '#A23B72'   # 紫红色
    color_best = '#F18F01'   # 橙色

    # 绘制训练集和测试集准确率，使用更粗的线条和更大的标记
    ax.plot(learning_rates, train_accs, marker='o', linewidth=2.5, markersize=8,
            label='Training Accuracy', color=color_train, alpha=0.8)
    ax.plot(learning_rates, test_accs, marker='s', linewidth=2.5, markersize=8,
            label='Test Accuracy', color=color_test, alpha=0.8)

    # 标注最佳值
    best_idx = np.argmax(test_accs)
    best_param = learning_rates[best_idx]
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
    ax.set_xlabel('Learning Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Learning Rate Ablation Study - speech_data_name', fontsize=16, fontweight='bold', pad=20)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)

    # 设置x轴刻度为对数
    ax.set_xscale('log')

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

    # 保存
    save_path = 'test_plot_learning_rate.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Test plot saved: {save_path}")
    print(f"  Best learning rate: {best_param}")
    print(f"  Best accuracy: {best_acc:.4f}")


def test_frame_shift_plot():
    """测试帧移对比图"""

    # 模拟数据 - 更多点，曲线更平滑
    frame_shifts = [3, 5, 7, 8, 10, 12, 15, 18, 20, 25, 30]
    train_accs = [0.85, 0.88, 0.91, 0.93, 0.95, 0.96, 0.95, 0.93, 0.90, 0.87, 0.83]
    test_accs = [0.80, 0.84, 0.87, 0.89, 0.91, 0.90, 0.88, 0.86, 0.84, 0.81, 0.77]

    fig, ax = plt.subplots(figsize=(14, 7))

    color_train = '#2E86AB'
    color_test = '#A23B72'
    color_best = '#F18F01'

    ax.plot(frame_shifts, train_accs, marker='o', linewidth=2.5, markersize=8,
            label='Training Accuracy', color=color_train, alpha=0.8)
    ax.plot(frame_shifts, test_accs, marker='s', linewidth=2.5, markersize=8,
            label='Test Accuracy', color=color_test, alpha=0.8)

    best_idx = np.argmax(test_accs)
    best_param = frame_shifts[best_idx]
    best_acc = test_accs[best_idx]

    ax.axvline(best_param, color=color_best, linestyle='--', alpha=0.6,
               linewidth=2, label=f'Best Value: {best_param}')
    ax.scatter([best_param], [best_acc], color=color_best, s=400,
               zorder=5, marker='*', edgecolors='white', linewidths=2)

    bbox_props = dict(boxstyle='round,pad=0.5', facecolor=color_best,
                      alpha=0.8, edgecolor='white', linewidth=2)
    ax.text(best_param, best_acc + 0.02, f'Best: {best_acc:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color='white', bbox=bbox_props)

    ax.set_xlabel('Frame Shift (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Frame Shift Ablation Study - speech_data_name', fontsize=16, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)

    legend = ax.legend(fontsize=12, loc='best', framealpha=0.95,
                       edgecolor='gray', fancybox=True, shadow=True)

    y_min = min(min(train_accs), min(test_accs))
    y_max = max(max(train_accs), max(test_accs))
    y_range = y_max - y_min
    ax.set_ylim([max(0, y_min - 0.05 * y_range), min(1, y_max + 0.1 * y_range)])

    ax.tick_params(axis='both', labelsize=11)

    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    save_path = 'test_plot_frame_shift.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Test plot saved: {save_path}")
    print(f"  Best frame shift: {best_param} ms")
    print(f"  Best accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    print("="*60)
    print("Testing Enhanced Plot Styles")
    print("="*60)
    print()

    test_plot()
    print()
    test_frame_shift_plot()

    print()
    print("="*60)
    print("✓ All test plots generated successfully!")
    print("="*60)
