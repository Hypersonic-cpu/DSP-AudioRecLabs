"""
完整实验流程：语音识别实验
"""
import os
import sys
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.time_domain.audio_processing import process_audio_file, load_wav, preprocess
from src.time_domain.feature_extraction import (
    extract_features_from_frames, normalize_features, extract_frame_features
)
from src.time_domain.models import create_classifier
from src.time_domain.visualization import (
    plot_waveform, plot_endpoint_detection, plot_frame_features,
    plot_confusion_matrix, plot_classifier_comparison, plot_window_comparison,
    plot_mlp_training_history, plot_feature_distribution
)


class SpeechRecognitionExperiment:
    """语音识别实验主类"""

    def __init__(self, data_dir, results_dir):
        """
        Args:
            data_dir: 数据目录
            results_dir: 结果保存目录
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.class_names = None
        self.X = None
        self.y = None
        self.feature_names = None

    def load_dataset(self, window_type='hamming', do_endpoint_detection=True):
        """
        加载数据集并提取特征

        Args:
            window_type: 窗函数类型
            do_endpoint_detection: 是否进行端点检测

        Returns:
            X: 特征矩阵
            y: 标签数组
            feature_names: 特征名称列表
        """
        print(f"\n{'='*60}")
        print(f"Loading dataset with window type: {window_type}")
        print(f"Endpoint detection: {do_endpoint_detection}")
        print(f"{'='*60}\n")

        # 获取所有类别文件夹
        class_folders = sorted([d for d in os.listdir(self.data_dir)
                               if os.path.isdir(os.path.join(self.data_dir, d))])

        # 过滤掉隐藏文件夹
        class_folders = [f for f in class_folders if not f.startswith('.')]

        print(f"Found {len(class_folders)} classes: {class_folders}\n")

        self.class_names = class_folders

        all_features = []
        all_labels = []

        # 遍历每个类别
        for class_idx, class_name in enumerate(tqdm(class_folders, desc="Processing classes")):
            class_path = os.path.join(self.data_dir, class_name)

            # 获取该类别下的所有WAV文件
            wav_files = glob(os.path.join(class_path, '*.wav'))

            print(f"  Class '{class_name}': {len(wav_files)} samples")

            # 处理每个音频文件
            for wav_file in wav_files:
                try:
                    # 处理音频文件
                    frames, sample_rate, metadata = process_audio_file(
                        wav_file,
                        frame_length=config.FRAME_LENGTH,
                        frame_shift=config.FRAME_SHIFT,
                        window_type=window_type,
                        do_endpoint_detection=do_endpoint_detection,
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
                    print(f"    Error processing {wav_file}: {e}")
                    continue

        # 转换为numpy数组
        X = np.array(all_features)
        y = np.array(all_labels)

        print(f"\nDataset loaded successfully!")
        print(f"Total samples: {len(X)}")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Class distribution: {np.bincount(y)}")

        self.X = X
        self.y = y
        self.feature_names = feature_names

        return X, y, feature_names

    def visualize_sample(self, sample_idx=0, window_type='hamming'):
        """
        可视化一个样本的处理过程

        Args:
            sample_idx: 样本索引（类别索引）
            window_type: 窗函数类型
        """
        # 如果class_names未初始化，先获取类别列表
        if self.class_names is None:
            class_folders = sorted([d for d in os.listdir(self.data_dir)
                                   if os.path.isdir(os.path.join(self.data_dir, d))
                                   and not d.startswith('.')])
            self.class_names = class_folders

        # 验证sample_idx有效性
        if sample_idx >= len(self.class_names):
            print(f"Error: sample_idx {sample_idx} out of range (max: {len(self.class_names)-1})")
            return

        print(f"\n{'='*60}")
        print(f"Visualizing sample from class: {self.class_names[sample_idx]}")
        print(f"{'='*60}\n")

        class_path = os.path.join(self.data_dir, self.class_names[sample_idx])
        wav_files = glob(os.path.join(class_path, '*.wav'))

        if len(wav_files) == 0:
            print("No samples found!")
            return

        # 选择第一个样本
        wav_file = wav_files[0]
        print(f"Processing: {os.path.basename(wav_file)}")

        from src.time_domain.audio_processing import load_wav, preprocess
        audio_data, sample_rate = load_wav(wav_file)
        audio_data = preprocess(audio_data)

        # 端点检测
        frames, _, metadata = process_audio_file(
            wav_file,
            frame_length=config.FRAME_LENGTH,
            frame_shift=config.FRAME_SHIFT,
            window_type=window_type,
            do_endpoint_detection=True
        )

        from src.time_domain.feature_extraction import extract_frame_features
        frame_features = extract_frame_features(frames)

        # 可视化
        vis_dir = os.path.join(self.results_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # 1. 原始波形
        plot_waveform(
            audio_data, sample_rate,
            title=f"Waveform - {self.class_names[sample_idx]}",
            save_path=os.path.join(vis_dir, f'waveform_{sample_idx}.png')
        )

        # 2. 端点检测
        plot_endpoint_detection(
            audio_data, sample_rate,
            metadata['start_point'], metadata['end_point'],
            metadata['energy_list'], metadata['zcr_list'],
            config.FRAME_SHIFT,
            title=f"Endpoint Detection - {self.class_names[sample_idx]}",
            save_path=os.path.join(vis_dir, f'endpoint_{sample_idx}.png')
        )

        # 3. 帧级特征
        plot_frame_features(
            frame_features,
            title=f"Frame Features - {self.class_names[sample_idx]}",
            save_path=os.path.join(vis_dir, f'features_{sample_idx}.png')
        )

        print(f"Visualizations saved to: {vis_dir}\n")

    def train_and_evaluate_classifier(self, classifier_type, X_train, X_test,
                                     y_train, y_test, **kwargs):
        """
        训练并评估分类器

        Args:
            classifier_type: 分类器类型
            X_train, X_test, y_train, y_test: 训练/测试数据
            **kwargs: 分类器参数

        Returns:
            results: 评估结果字典
        """
        print(f"Training {classifier_type}...")

        # 创建分类器
        if classifier_type == 'mlp':
            classifier = create_classifier(
                'mlp',
                input_size=X_train.shape[1],
                hidden_layers=config.MLP_HIDDEN_LAYERS,
                num_classes=len(self.class_names),
                learning_rate=config.MLP_LEARNING_RATE,
                epochs=config.MLP_EPOCHS,
                batch_size=config.MLP_BATCH_SIZE
            )
            classifier.fit(X_train, y_train, verbose=False)
        else:
            classifier = create_classifier(classifier_type, **kwargs)
            classifier.fit(X_train, y_train)

        # 评估
        results = classifier.evaluate(X_test, y_test)

        print(f"  Accuracy: {results['accuracy']:.4f}")

        return results

    def experiment_classifier_comparison(self, window_type='hamming'):
        """
        实验1：比较不同分类器

        Args:
            window_type: 窗函数类型
        """
        print(f"\n{'='*60}")
        print("EXPERIMENT 1: Classifier Comparison")
        print(f"{'='*60}\n")

        # 加载数据
        if self.X is None:
            self.load_dataset(window_type=window_type)

        # 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_SEED,
            stratify=self.y
        )

        # 特征归一化
        X_train, mean, std = normalize_features(X_train)
        X_test, _, _ = normalize_features(X_test, mean, std)

        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")

        # 定义要比较的分类器
        classifiers = {
            'KNN': ('knn', {'n_neighbors': config.KNN_N_NEIGHBORS}),
            'Naive Bayes': ('naive_bayes', {}),
            'Decision Tree': ('decision_tree', {}),
            'SVM': ('svm', {'C': config.SVM_C, 'kernel': config.SVM_KERNEL}),
            'MLP': ('mlp', {})
        }

        results_dict = {}

        # 训练并评估每个分类器
        for name, (clf_type, params) in classifiers.items():
            results = self.train_and_evaluate_classifier(
                clf_type, X_train, X_test, y_train, y_test, **params
            )
            results_dict[name] = results

        # 可视化结果
        exp_dir = os.path.join(self.results_dir, 'exp1_classifier_comparison')
        os.makedirs(exp_dir, exist_ok=True)

        # 1. 准确率对比
        plot_classifier_comparison(
            results_dict, metric='accuracy',
            title=f"Classifier Comparison (Window: {window_type})",
            save_path=os.path.join(exp_dir, 'accuracy_comparison.png')
        )

        # 2. 每个分类器的混淆矩阵
        for name, results in results_dict.items():
            plot_confusion_matrix(
                results['confusion_matrix'],
                self.class_names,
                title=f"Confusion Matrix - {name}",
                save_path=os.path.join(exp_dir, f'confusion_matrix_{name.replace(" ", "_")}.png')
            )

        # 3. MLP训练历史
        if 'MLP' in results_dict and 'train_losses' in results_dict['MLP']:
            plot_mlp_training_history(
                results_dict['MLP']['train_losses'],
                results_dict['MLP']['train_accuracies'],
                title="MLP Training History",
                save_path=os.path.join(exp_dir, 'mlp_training_history.png')
            )

        # 保存结果摘要
        self._save_results_summary(results_dict, exp_dir)

        print(f"\nResults saved to: {exp_dir}\n")

        return results_dict

    def experiment_window_comparison(self):
        """
        实验2：比较不同窗函数
        """
        print(f"\n{'='*60}")
        print("EXPERIMENT 2: Window Function Comparison")
        print(f"{'='*60}\n")

        window_results = {}

        # 对每种窗函数进行实验
        for window_type in config.WINDOW_TYPES:
            print(f"\n--- Window Type: {window_type} ---")

            # 加载数据
            self.load_dataset(window_type=window_type)

            # 划分训练/测试集
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y,
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_SEED,
                stratify=self.y
            )

            # 特征归一化
            X_train, mean, std = normalize_features(X_train)
            X_test, _, _ = normalize_features(X_test, mean, std)

            # 测试几个主要分类器
            classifiers = {
                'KNN': ('knn', {'n_neighbors': config.KNN_N_NEIGHBORS}),
                'SVM': ('svm', {'C': config.SVM_C}),
                'MLP': ('mlp', {})
            }

            classifier_results = {}

            for name, (clf_type, params) in classifiers.items():
                results = self.train_and_evaluate_classifier(
                    clf_type, X_train, X_test, y_train, y_test, **params
                )
                classifier_results[name] = results

            window_results[window_type] = classifier_results

        # 可视化对比
        exp_dir = os.path.join(self.results_dir, 'exp2_window_comparison')
        os.makedirs(exp_dir, exist_ok=True)

        plot_window_comparison(
            window_results, metric='accuracy',
            title="Window Function Comparison",
            save_path=os.path.join(exp_dir, 'window_comparison.png')
        )

        # 保存详细结果
        self._save_window_comparison_summary(window_results, exp_dir)

        print(f"\nResults saved to: {exp_dir}\n")

        return window_results

    def experiment_feature_analysis(self, window_type='hamming'):
        """
        实验3：特征分析

        Args:
            window_type: 窗函数类型
        """
        print(f"\n{'='*60}")
        print("EXPERIMENT 3: Feature Analysis")
        print(f"{'='*60}\n")

        # 加载数据
        if self.X is None:
            self.load_dataset(window_type=window_type)

        # 可视化特征分布
        exp_dir = os.path.join(self.results_dir, 'exp3_feature_analysis')
        os.makedirs(exp_dir, exist_ok=True)

        plot_feature_distribution(
            self.X, self.y, self.feature_names,
            title="Feature Distribution by Class",
            save_path=os.path.join(exp_dir, 'feature_distribution.png')
        )

        # 特征统计
        self._analyze_features(exp_dir)

        print(f"\nResults saved to: {exp_dir}\n")

    def _save_results_summary(self, results_dict, save_dir):
        """保存结果摘要到文本文件"""
        summary_path = os.path.join(save_dir, 'results_summary.txt')

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("CLASSIFIER COMPARISON RESULTS\n")
            f.write("="*60 + "\n\n")

            for name, results in results_dict.items():
                f.write(f"\n{name}:\n")
                f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"  Classification Report:\n")

                report = results['classification_report']
                for class_idx, class_name in enumerate(self.class_names):
                    if str(class_idx) in report:
                        metrics = report[str(class_idx)]
                        f.write(f"    {class_name}:\n")
                        f.write(f"      Precision: {metrics['precision']:.4f}\n")
                        f.write(f"      Recall: {metrics['recall']:.4f}\n")
                        f.write(f"      F1-score: {metrics['f1-score']:.4f}\n")

                f.write("\n" + "-"*60 + "\n")

        print(f"Summary saved to: {summary_path}")

    def _save_window_comparison_summary(self, window_results, save_dir):
        """保存窗函数对比结果"""
        summary_path = os.path.join(save_dir, 'window_comparison_summary.txt')

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("WINDOW FUNCTION COMPARISON RESULTS\n")
            f.write("="*60 + "\n\n")

            for window_type, classifier_results in window_results.items():
                f.write(f"\nWindow Type: {window_type}\n")
                f.write("-"*40 + "\n")

                for classifier_name, results in classifier_results.items():
                    f.write(f"  {classifier_name}: {results['accuracy']:.4f}\n")

                f.write("\n")

        print(f"Summary saved to: {summary_path}")

    def _analyze_features(self, save_dir):
        """分析特征统计信息"""
        analysis_path = os.path.join(save_dir, 'feature_analysis.txt')

        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("FEATURE ANALYSIS\n")
            f.write("="*60 + "\n\n")

            f.write(f"Total samples: {len(self.X)}\n")
            f.write(f"Feature dimension: {self.X.shape[1]}\n")
            f.write(f"Number of classes: {len(self.class_names)}\n\n")

            f.write("Feature Statistics:\n")
            f.write("-"*40 + "\n")

            for i, feature_name in enumerate(self.feature_names):
                feature_data = self.X[:, i]
                f.write(f"\n{feature_name}:\n")
                f.write(f"  Mean: {np.mean(feature_data):.6f}\n")
                f.write(f"  Std: {np.std(feature_data):.6f}\n")
                f.write(f"  Min: {np.min(feature_data):.6f}\n")
                f.write(f"  Max: {np.max(feature_data):.6f}\n")

        print(f"Feature analysis saved to: {analysis_path}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("SPEECH RECOGNITION EXPERIMENT")
    print("Based on Time-Domain Analysis")
    print("="*60 + "\n")

    # 创建实验对象
    experiment = SpeechRecognitionExperiment(
        data_dir=config.DATA_DIR,
        results_dir=config.RESULTS_DIR
    )

    # 可视化样本（每个数字各一个）
    print("Step 1: Visualizing sample processing...")
    for i in range(min(3, len(experiment.class_names) if experiment.class_names else 3)):
        experiment.visualize_sample(sample_idx=i, window_type='hamming')

    # 实验1：分类器对比
    print("\nStep 2: Running classifier comparison experiment...")
    experiment.experiment_classifier_comparison(window_type='hamming')

    # 实验2：窗函数对比
    print("\nStep 3: Running window function comparison experiment...")
    experiment.experiment_window_comparison()

    # 实验3：特征分析
    print("\nStep 4: Running feature analysis...")
    experiment.experiment_feature_analysis(window_type='hamming')

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
