#!/usr/bin/env python3
import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='语音识别实验系统')

    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='数据集目录路径（例如：~/Downloads/speech_data 或 ./data）'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='结果保存目录（默认：./results）'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        default='all',
        choices=['all', 'classifier', 'window', 'feature', 'visualize'],
        help='要运行的实验：all（全部）, classifier（分类器对比）, window（窗函数对比）, feature（特征分析）, visualize（样本可视化）'
    )

    parser.add_argument(
        '--window-type',
        type=str,
        default='hamming',
        choices=['rectangular', 'hamming', 'hanning'],
        help='窗函数类型（默认：hamming）'
    )

    args = parser.parse_args()

    # 设置数据路径
    if args.data_dir:
        data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
        os.environ['SPEECH_DATA_DIR'] = data_dir
        print(f"数据目录设置为: {data_dir}")

    # 导入实验模块
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config
    from experiments.run_experiments import SpeechRecognitionExperiment

    # 检查数据目录
    if not os.path.exists(config.DATA_DIR):
        print(f"\n❌ 错误：数据目录不存在: {config.DATA_DIR}")
        print("\n请使用以下方式之一指定数据路径：")
        print("  1. 命令行参数: python run.py --data-dir ~/Downloads/speech_data")
        print("  2. 修改 config.py 中的 DATA_DIR 变量")
        print("  3. 设置环境变量: export SPEECH_DATA_DIR=/path/to/data\n")
        sys.exit(1)

    # 检查数据格式
    subdirs = [d for d in os.listdir(config.DATA_DIR)
               if os.path.isdir(os.path.join(config.DATA_DIR, d)) and not d.startswith('.')]

    if len(subdirs) == 0:
        print(f"\n❌ 错误：数据目录 {config.DATA_DIR} 中没有找到类别子文件夹")
        print("\n数据应该按以下结构组织：")
        print("  data/")
        print("    ├── 0/")
        print("    │   ├── sample1.wav")
        print("    │   └── sample2.wav")
        print("    ├── 1/")
        print("    │   └── ...")
        print("    └── ...\n")
        sys.exit(1)

    print(f"\n✓ 找到 {len(subdirs)} 个类别: {subdirs}\n")

    # 创建实验对象
    experiment = SpeechRecognitionExperiment(
        data_dir=config.DATA_DIR,
        results_dir=config.RESULTS_DIR
    )

    print("="*60)
    print("语音识别实验系统")
    print("基于时域分析技术")
    print("="*60)

    # 根据参数运行实验
    if args.experiment == 'all':
        print("\n运行所有实验...\n")

        # 可视化样本
        print("1. 可视化样本处理过程...")
        for i in range(min(3, len(subdirs))):
            experiment.visualize_sample(sample_idx=i, window_type=args.window_type)

        # 分类器对比
        print("\n2. 分类器对比实验...")
        experiment.experiment_classifier_comparison(window_type=args.window_type)

        # 窗函数对比
        print("\n3. 窗函数对比实验...")
        experiment.experiment_window_comparison()

        # 特征分析
        print("\n4. 特征分析...")
        experiment.experiment_feature_analysis(window_type=args.window_type)

    elif args.experiment == 'visualize':
        print("\n可视化样本...")
        for i in range(min(3, len(subdirs))):
            experiment.visualize_sample(sample_idx=i, window_type=args.window_type)

    elif args.experiment == 'classifier':
        print("\n运行分类器对比实验...")
        experiment.experiment_classifier_comparison(window_type=args.window_type)

    elif args.experiment == 'window':
        print("\n运行窗函数对比实验...")
        experiment.experiment_window_comparison()

    elif args.experiment == 'feature':
        print("\n运行特征分析...")
        experiment.experiment_feature_analysis(window_type=args.window_type)

    print("\n" + "="*60)
    print("✓ 实验完成！")
    print(f"结果已保存到: {config.RESULTS_DIR}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
