#!/usr/bin/env python3
"""
Frequency Domain Speech Recognition Experiment Runner.
Implements Experiment 2 (MFCC) and Experiment 3 (DTW Recognition).

Usage:
    python run_freq_domain.py --mode train
    python run_freq_domain.py --mode evaluate
    python run_freq_domain.py --mode recognize --file path/to/audio.wav
    python run_freq_domain.py --mode full
"""
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.freq_domain import (
    split_dataset, train_templates_from_files,
    save_templates, load_templates,
    recognize_file, evaluate_files,
    generate_all_visualizations,
    extract_mfcc, extract_mfcc_with_delta,
    extract_gfcc, extract_gfcc_with_delta
)
from src.freq_domain.dtw import dtw_distance
from src.common.evaluation import confusion_matrix_report, print_evaluation_report
from src.common.audio_io import load_audio
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm


def get_model_path():
    """Get default model save path."""
    return os.path.join(config.RESULTS_DIR, 'freq_domain', 'temlates.pt')


def get_split_path():
    """Get default split info save path."""
    return os.path.join(config.RESULTS_DIR, 'freq_domain', 'data_split.pt')


def mode_train(args):
    """Train MFCC templates from data."""
    print("\n" + "=" * 60)
    print("MODE: TRAIN - Building MFCC Templates")
    print("=" * 60)

    data_dir = args.data_dir or config.DATA_DIR
    model_path = args.model or get_model_path()
    split_path = get_split_path()

    print(f"\nData directory: {data_dir}")
    print(f"Model save path: {model_path}")
    print(f"Sample rate: {config.SAMPLE_RATE}")
    print(f"N_MFCC: {args.n_mfcc}")
    print(f"Test size: {args.test_size}")
    print(f"Template strategy: {args.template_strategy}")

    # Check data directory
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory not found: {data_dir}")
        return

    # Split dataset
    print(f"\nSplitting dataset (test_size={args.test_size})...")
    train_files, test_files = split_dataset(
        data_dir, test_size=args.test_size, random_state=config.RANDOM_SEED
    )

    # Print split info
    total_train = sum(len(f) for f in train_files.values())
    total_test = sum(len(f) for f in test_files.values())
    print(f"Train samples: {total_train}, Test samples: {total_test}")

    # Train templates from train set only
    templates = train_templates_from_files(
        train_files=train_files,
        sample_rate=config.SAMPLE_RATE,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        frame_len_ms=config.FRAME_LENGTH_MS,
        frame_shift_ms=config.FRAME_SHIFT_MS,
        template_strategy=args.template_strategy
    )

    # Save templates and split info
    if templates:
        save_templates(templates, model_path)

        # Save split info for evaluation
        import torch
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        torch.save({'train': train_files, 'test': test_files}, split_path)
        print(f"Data split saved to: {split_path}")

        print(f"\nTraining completed. {len(templates)} templates saved.")
    else:
        print("\nError: No templates were generated.")


def mode_evaluate(args):
    """Evaluate recognition accuracy on test set."""
    print("\n" + "=" * 60)
    print("MODE: EVALUATE - Testing Recognition Accuracy")
    print("=" * 60)

    model_path = args.model or get_model_path()
    split_path = get_split_path()

    print(f"\nModel path: {model_path}")

    # Load templates
    try:
        templates = load_templates(model_path)
    except FileNotFoundError:
        print(f"\nError: Model not found at {model_path}")
        print("Please run training first: python run_freq_domain.py --mode train")
        return

    # Load split info
    import torch
    try:
        split_info = torch.load(split_path)
        test_files = split_info['test']
        print(f"Loaded test split from: {split_path}")
    except FileNotFoundError:
        print(f"\nError: Split info not found at {split_path}")
        print("Please run training first to generate data split.")
        return

    # Evaluate on test set only
    y_true, y_pred = evaluate_files(
        test_files=test_files,
        templates=templates,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        frame_len_ms=config.FRAME_LENGTH_MS,
        frame_shift_ms=config.FRAME_SHIFT_MS
    )

    # Generate and print report
    class_names = [str(i) for i in range(10)]
    results = confusion_matrix_report(y_true, y_pred, class_names)
    print_evaluation_report(results, class_names)

    # Save results
    results_dir = os.path.join(config.RESULTS_DIR, 'freq_domain')
    os.makedirs(results_dir, exist_ok=True)

    # Save confusion matrix
    import numpy as np
    np.savetxt(
        os.path.join(results_dir, 'confusion_matrix.txt'),
        results['confusion_matrix'],
        fmt='%d'
    )

    # Save summary
    with open(os.path.join(results_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Total test samples: {len(y_true)}\n")
        f.write(f"Correct predictions: {sum(y_true == y_pred)}\n")

    print(f"\nResults saved to: {results_dir}")


def mode_recognize(args):
    """Recognize a single audio file."""
    print("\n" + "=" * 60)
    print("MODE: RECOGNIZE - Single File Recognition")
    print("=" * 60)

    if not args.file:
        print("\nError: Please specify audio file with --file")
        return

    if not os.path.exists(args.file):
        print(f"\nError: File not found: {args.file}")
        return

    model_path = args.model or get_model_path()

    # Load templates
    try:
        templates = load_templates(model_path)
    except FileNotFoundError:
        print(f"\nError: Model not found at {model_path}")
        return

    # Recognize
    print(f"\nRecognizing: {args.file}")
    recognized, distances = recognize_file(
        args.file, templates,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        frame_len_ms=config.FRAME_LENGTH_MS,
        frame_shift_ms=config.FRAME_SHIFT_MS,
        verbose=True
    )

    print(f"\n{'=' * 40}")
    print(f"RECOGNIZED DIGIT: {recognized}")
    print(f"{'=' * 40}")


def mode_visualize(args):
    """Generate visualizations for experiment report."""
    print("\n" + "=" * 60)
    print("MODE: VISUALIZE - Generating Figures")
    print("=" * 60)

    model_path = args.model or get_model_path()
    split_path = get_split_path()

    # Load templates
    try:
        templates = load_templates(model_path)
    except FileNotFoundError:
        print(f"\nError: Model not found at {model_path}")
        print("Please run training first.")
        return

    # Load split info and get a sample
    import torch
    try:
        split_info = torch.load(split_path)
        test_files = split_info['test']
    except FileNotFoundError:
        print(f"\nError: Split info not found. Please run training first.")
        return

    # Get a correctly recognized sample for visualization
    sample_digit = None
    sample_file = None
    for digit, files in sorted(test_files.items(), key=lambda kv: int(kv[0])):
        for wav_path in files:
            try:
                recognized, _ = recognize_file(
                    wav_path, templates,
                    n_mfcc=args.n_mfcc,
                    n_fft=args.n_fft,
                    frame_len_ms=config.FRAME_LENGTH_MS,
                    frame_shift_ms=config.FRAME_SHIFT_MS,
                    verbose=False
                )
                if str(recognized) == str(digit):
                    sample_digit = str(digit)
                    sample_file = wav_path
                    break
            except Exception:
                continue
        if sample_file:
            break

    # Fallback to any available sample if none were correctly recognized
    if sample_file is None:
        fallback_digit = '0'
        if test_files.get(fallback_digit):
            sample_file = test_files[fallback_digit][0]
            sample_digit = fallback_digit
        else:
            for digit, files in test_files.items():
                if files:
                    sample_file = files[0]
                    sample_digit = digit
                    break
            else:
                print("Error: No test samples found.")
                return

    signal, sample_rate = load_audio(sample_file)
    print(f"Using sample: {os.path.basename(sample_file)} (digit {sample_digit})")

    # Run evaluation for results
    print("\nRunning evaluation for visualization data...")
    y_true, y_pred = evaluate_files(
        test_files=test_files,
        templates=templates,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        frame_len_ms=config.FRAME_LENGTH_MS,
        frame_shift_ms=config.FRAME_SHIFT_MS
    )

    # Generate report
    class_names = [str(i) for i in range(10)]
    results = confusion_matrix_report(y_true, y_pred, class_names)

    # Generate all visualizations
    output_dir = os.path.join(config.RESULTS_DIR, 'freq_domain', 'figures')
    generate_all_visualizations(
        signal=signal,
        sample_rate=sample_rate,
        templates=templates,
        results=results,
        output_dir=output_dir,
        digit_label=sample_digit
    )

    print(f"\nAll visualizations saved to: {output_dir}")


def mode_full(args):
    """Run complete experiment: train + evaluate + visualize."""
    print("\n" + "=" * 60)
    print("MODE: FULL - Complete Experiment Pipeline")
    print("=" * 60)

    # Step 1: Train
    print("\n[Step 1/3] Training templates...")
    mode_train(args)

    # Step 2: Evaluate
    print("\n[Step 2/3] Evaluating on dataset...")
    mode_evaluate(args)

    # Step 3: Visualize
    print("\n[Step 3/3] Generating visualizations...")
    mode_visualize(args)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)


def train_templates_with_feature(train_files, feature_type, n_coef, n_fft,
                                  frame_len_ms, frame_shift_ms, template_strategy='first'):
    """
    Train templates using specified feature extraction method.

    Args:
        train_files: Dictionary {digit: [file_paths]}
        feature_type: 'mfcc', 'mfcc_delta', 'mfcc_delta_delta', 'gfcc'
        n_coef: Number of base coefficients
        n_fft: FFT size
        frame_len_ms: Frame length in ms
        frame_shift_ms: Frame shift in ms
        template_strategy: 'first' or 'mean'

    Returns:
        templates: Dictionary {digit: feature_tensor}
    """
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    templates = {}

    for digit, files in train_files.items():
        if not files:
            continue

        features_list = []
        for filepath in files:
            signal, sr = load_audio(filepath)

            # Endpoint detection and ROI extraction + average-energy normalization
            try:
                from src.time_domain.audio_processing import endpoint_detection
                start_pt, end_pt, _, _ = endpoint_detection(
                    signal,
                    frame_length=config.FRAME_LENGTH,
                    frame_shift=config.FRAME_SHIFT,
                    energy_high_ratio=config.ENERGY_HIGH_RATIO,
                    energy_low_ratio=config.ENERGY_LOW_RATIO,
                    zcr_threshold_ratio=config.ZCR_THRESHOLD_RATIO
                )
                signal_roi = signal[start_pt:end_pt]
                if len(signal_roi) == 0:
                    signal_roi = signal
            except Exception:
                signal_roi = signal

            try:
                mean_energy = np.mean(np.array(signal_roi) ** 2)
                if mean_energy > 0:
                    signal_roi = signal_roi / float(np.sqrt(mean_energy))
            except Exception:
                pass

            # Extract features based on type
            if feature_type == 'mfcc':
                feat = extract_mfcc(
                    signal_roi, sr, n_mfcc=n_coef, n_fft=n_fft,
                    frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
                    device=device
                )
            elif feature_type == 'mfcc_delta':
                feat = extract_mfcc_with_delta(
                    signal, sr, n_mfcc=n_coef, n_fft=n_fft,
                    frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
                    include_delta=True, include_delta_delta=False,
                    device=device
                )
            elif feature_type == 'mfcc_delta_delta':
                feat = extract_mfcc_with_delta(
                    signal, sr, n_mfcc=n_coef, n_fft=n_fft,
                    frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
                    include_delta=True, include_delta_delta=True,
                    device=device
                )
            elif feature_type == 'gfcc':
                feat = extract_gfcc(
                    signal, sr, n_gfcc=n_coef, n_fft=n_fft,
                    frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
                    device=device
                )
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            features_list.append(feat)

        # Create template based on strategy
        if template_strategy == 'first':
            templates[digit] = features_list[0]
        else:  # mean
            # DTW average would be complex, so just use first for now
            templates[digit] = features_list[0]

    return templates


def evaluate_with_feature(test_files, templates, feature_type, n_coef, n_fft,
                          frame_len_ms, frame_shift_ms):
    """
    Evaluate recognition using specified feature extraction method.

    Returns:
        y_true, y_pred: Arrays of true and predicted labels
    """
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    y_true = []
    y_pred = []

    total_files = sum(len(files) for files in test_files.values())

    for digit_str, wav_files in test_files.items():
        digit = int(digit_str)

        for wav_file in tqdm(wav_files, desc=f"Digit {digit}", leave=False):
            try:
                signal, sr = load_audio(wav_file)

                # Extract features based on type
                if feature_type == 'mfcc':
                    feat = extract_mfcc(
                        signal, sr, n_mfcc=n_coef, n_fft=n_fft,
                        frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
                        device=device
                    )
                elif feature_type == 'mfcc_delta':
                    feat = extract_mfcc_with_delta(
                        signal, sr, n_mfcc=n_coef, n_fft=n_fft,
                        frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
                        include_delta=True, include_delta_delta=False,
                        device=device
                    )
                elif feature_type == 'mfcc_delta_delta':
                    feat = extract_mfcc_with_delta(
                        signal, sr, n_mfcc=n_coef, n_fft=n_fft,
                        frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
                        include_delta=True, include_delta_delta=True,
                        device=device
                    )
                elif feature_type == 'gfcc':
                    feat = extract_gfcc(
                        signal, sr, n_gfcc=n_coef, n_fft=n_fft,
                        frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
                        device=device
                    )
                else:
                    raise ValueError(f"Unknown feature type: {feature_type}")

                # Find best matching template
                distances = {}
                for template_digit, template in templates.items():
                    if not torch.is_tensor(template):
                        template = torch.from_numpy(template).float()
                    template = template.to(device)
                    feat = feat.to(device)

                    dist = dtw_distance(feat, template)
                    distances[template_digit] = dist

                recognized = min(distances, key=distances.get)
                y_true.append(digit)
                y_pred.append(int(recognized))

            except Exception as e:
                print(f"Error processing {wav_file}: {e}")

    return np.array(y_true), np.array(y_pred)


def mode_ablation(args):
    """Run ablation experiments comparing different feature types."""
    print("\n" + "=" * 60)
    print("MODE: ABLATION - Feature Comparison Experiments")
    print("=" * 60)

    data_dir = args.data_dir or config.DATA_DIR
    results_dir = os.path.join(config.RESULTS_DIR, 'freq_domain', 'ablation')
    os.makedirs(results_dir, exist_ok=True)

    # Ablation configurations
    configs = [
        {'name': 'MFCC-13D', 'type': 'mfcc', 'dim': 13},
        {'name': 'MFCC+Delta-26D', 'type': 'mfcc_delta', 'dim': 26},
        {'name': 'MFCC+Delta+DeltaDelta-39D', 'type': 'mfcc_delta_delta', 'dim': 39},
        {'name': 'GFCC-13D', 'type': 'gfcc', 'dim': 13},
    ]

    # Split dataset once
    print(f"\nSplitting dataset (test_size={args.test_size})...")
    train_files, test_files = split_dataset(
        data_dir, test_size=args.test_size, random_state=config.RANDOM_SEED
    )

    total_train = sum(len(f) for f in train_files.values())
    total_test = sum(len(f) for f in test_files.values())
    print(f"Train samples: {total_train}, Test samples: {total_test}")

    # Store results
    all_results = []
    class_names = [str(i) for i in range(10)]

    # Log file for paper tables
    log_path = os.path.join(results_dir, 'ablation_results.log')

    with open(log_path, 'w') as log_file:
        # Write header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"# Ablation Experiment Results\n")
        log_file.write(f"# Generated: {timestamp}\n")
        log_file.write(f"# Train samples: {total_train}, Test samples: {total_test}\n")
        log_file.write(f"# Test size: {args.test_size}\n\n")

        # Run each configuration
        for cfg in configs:
            print(f"\n{'=' * 50}")
            print(f"Testing: {cfg['name']} (Dimension: {cfg['dim']})")
            print(f"{'=' * 50}")

            # Train templates
            print("Training templates...")
            templates = train_templates_with_feature(
                train_files=train_files,
                feature_type=cfg['type'],
                n_coef=args.n_mfcc,
                n_fft=args.n_fft,
                frame_len_ms=config.FRAME_LENGTH_MS,
                frame_shift_ms=config.FRAME_SHIFT_MS,
                template_strategy=args.template_strategy
            )

            # Evaluate
            print("Evaluating...")
            y_true, y_pred = evaluate_with_feature(
                test_files=test_files,
                templates=templates,
                feature_type=cfg['type'],
                n_coef=args.n_mfcc,
                n_fft=args.n_fft,
                frame_len_ms=config.FRAME_LENGTH_MS,
                frame_shift_ms=config.FRAME_SHIFT_MS
            )

            # Calculate metrics
            results = confusion_matrix_report(y_true, y_pred, class_names)
            accuracy = results['accuracy']

            # Store result
            all_results.append({
                'name': cfg['name'],
                'type': cfg['type'],
                'dim': cfg['dim'],
                'accuracy': accuracy,
                'results': results
            })

            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

            # Write to log
            log_file.write(f"\n## {cfg['name']}\n")
            log_file.write(f"Feature Type: {cfg['type']}\n")
            log_file.write(f"Dimension: {cfg['dim']}\n")
            log_file.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

            # Per-class metrics
            log_file.write("\nPer-class Precision/Recall/F1:\n")
            log_file.write("| Digit | Precision | Recall | F1-Score | Support |\n")
            log_file.write("|-------|-----------|--------|----------|--------|\n")

            report = results['classification_report']
            for name in class_names:
                if name in report:
                    metrics = report[name]
                    precision = metrics['precision']
                    recall = metrics['recall']
                    f1 = metrics['f1-score']
                    support = int(metrics['support'])
                    log_file.write(f"| {name} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support} |\n")

            log_file.write("\n")

        # Write summary table for paper
        log_file.write("\n" + "=" * 60 + "\n")
        log_file.write("# SUMMARY TABLE (for paper)\n")
        log_file.write("=" * 60 + "\n\n")

        log_file.write("| Method | Dimension | Accuracy (%) |\n")
        log_file.write("|--------|-----------|-------------|\n")

        for result in all_results:
            log_file.write(f"| {result['name']} | {result['dim']} | {result['accuracy']*100:.2f} |\n")

        # Find best result
        best = max(all_results, key=lambda x: x['accuracy'])
        log_file.write(f"\nBest Method: {best['name']} with {best['accuracy']*100:.2f}% accuracy\n")

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION EXPERIMENT SUMMARY")
    print("=" * 60)
    print("\n| Method                      | Dim | Accuracy |")
    print("|-----------------------------|----|----------|")

    for result in all_results:
        print(f"| {result['name']:<27} | {result['dim']:>2} | {result['accuracy']*100:>6.2f}% |")

    best = max(all_results, key=lambda x: x['accuracy'])
    print(f"\nBest: {best['name']} ({best['accuracy']*100:.2f}%)")
    print(f"\nResults saved to: {log_path}")

    # Generate ablation comparison visualization
    print("\nGenerating ablation comparison figures...")
    generate_ablation_figures(all_results, results_dir)


def generate_ablation_figures(all_results: list, output_dir: str):
    """
    Generate visualization figures for ablation experiments.

    Args:
        all_results: List of result dictionaries from ablation experiments
        output_dir: Directory to save figures
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set theme
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "savefig.bbox": "tight"
    })

    os.makedirs(output_dir, exist_ok=True)

    # 1. Bar chart comparing accuracy
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [r['name'] for r in all_results]
    accuracies = [r['accuracy'] * 100 for r in all_results]
    colors = sns.color_palette("viridis", len(names))

    bars = ax.bar(names, accuracies, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Feature Type', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Feature Comparison: Recognition Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])

    # Rotate x labels for better readability
    plt.xticks(rotation=15, ha='right')

    fig.tight_layout()
    save_path = os.path.join(output_dir, 'ablation_accuracy_comparison.png')
    fig.savefig(save_path, dpi=220, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)

    # 2. Grouped bar chart for per-class F1 scores
    fig, ax = plt.subplots(figsize=(14, 6))

    class_names = [str(i) for i in range(10)]
    x = np.arange(len(class_names))
    width = 0.2
    offsets = np.linspace(-1.5*width, 1.5*width, len(all_results))

    for idx, result in enumerate(all_results):
        f1_scores = []
        report = result['results']['classification_report']
        for name in class_names:
            if name in report:
                f1_scores.append(report[name]['f1-score'])
            else:
                f1_scores.append(0)

        ax.bar(x + offsets[idx], f1_scores, width,
               label=result['name'], color=colors[idx], edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Digit', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Per-Digit F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([0, 1.1])

    fig.tight_layout()
    save_path = os.path.join(output_dir, 'ablation_f1_comparison.png')
    fig.savefig(save_path, dpi=220, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)

    # 3. Dimension vs Accuracy scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))

    dims = [r['dim'] for r in all_results]
    accs = [r['accuracy'] * 100 for r in all_results]
    labels = [r['name'] for r in all_results]

    scatter = ax.scatter(dims, accs, s=200, c=colors, edgecolors='black', linewidth=1.5, zorder=5)

    # Add labels
    for i, (d, a, label) in enumerate(zip(dims, accs, labels)):
        ax.annotate(label, (d, a), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, ha='left')

    ax.set_xlabel('Feature Dimension', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Feature Dimension vs Recognition Accuracy', fontsize=14, fontweight='bold')

    fig.tight_layout()
    save_path = os.path.join(output_dir, 'ablation_dimension_accuracy.png')
    fig.savefig(save_path, dpi=220, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)

    print(f"All ablation figures saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Frequency Domain Speech Recognition (MFCC + DTW)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_freq_domain.py --mode train
  python run_freq_domain.py --mode evaluate
  python run_freq_domain.py --mode recognize --file test.wav
  python run_freq_domain.py --mode full
        """
    )

    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'recognize', 'visualize', 'full', 'ablation'],
                        help='Operation mode')

    # Paths
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory (default: from config)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model file path (default: results/freq_domain/templates.pt)')
    parser.add_argument('--file', type=str, default=None,
                        help='Audio file for recognition mode')

    # MFCC parameters
    parser.add_argument('--n-mfcc', type=int, default=13,
                        help='Number of MFCC coefficients (default: 13)')
    parser.add_argument('--n-fft', type=int, default=512,
                        help='FFT size (default: 512)')

    # Training parameters
    parser.add_argument('--template-strategy', type=str, default='first',
                        choices=['first', 'mean'],
                        help='Template selection strategy (default: first)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')

    args = parser.parse_args()

    # Route to appropriate mode
    if args.mode == 'train':
        mode_train(args)
    elif args.mode == 'evaluate':
        mode_evaluate(args)
    elif args.mode == 'recognize':
        mode_recognize(args)
    elif args.mode == 'visualize':
        mode_visualize(args)
    elif args.mode == 'full':
        mode_full(args)
    elif args.mode == 'ablation':
        mode_ablation(args)


if __name__ == '__main__':
    main()
