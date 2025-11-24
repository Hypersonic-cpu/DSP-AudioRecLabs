"""
Visualization module for frequency domain speech recognition.
All plots use English labels to avoid font issues.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

from .features import (
    pre_emphasis, frame_signal, apply_window,
    compute_power_spectrum, create_mel_filterbank, extract_mfcc
)

# Consistent visual theme for all plots
COLORS = {
    "primary": "#1f6feb",
    "accent": "#ff9f1c",
    "secondary": "#2a9d8f",
    "muted": "#5f6c7b",
    "background": "#f8fafc"
}
MFCC_CMAP = "coolwarm"

sns.set_theme(
    style="whitegrid",
    context="talk",
    rc={
        "axes.edgecolor": "#e5e7eb",
        "axes.facecolor": COLORS["background"],
        "figure.facecolor": "white",
        "grid.alpha": 0.25,
        "grid.linestyle": "--"
    }
)
sns.set_palette([COLORS["primary"], COLORS["accent"], COLORS["secondary"]])
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "savefig.bbox": "tight"
})


def _finalize_figure(fig, save_path: str = None, dpi: int = 220):
    """Tight layout, save if needed, and close."""
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def _add_zero_line(ax):
    """Add a thin zero-reference line to time-domain plots."""
    ax.axhline(0, color="#d0d7de", linewidth=0.8, linestyle="--", zorder=0)


def _zscore_for_display(array: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize an array for visualization and return symmetric vlim."""
    arr = array.astype(float)
    mean = arr.mean()
    std = arr.std() + 1e-9
    norm = (arr - mean) / std
    vlim = np.percentile(np.abs(norm), 97)
    return norm, vlim


def plot_waveform(signal: np.ndarray, sample_rate: int,
                  title: str = "Audio Waveform",
                  save_path: str = None, figsize: tuple = (12, 3)):
    """
    Plot audio waveform.

    Args:
        signal: Audio signal
        sample_rate: Sample rate in Hz
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    time = np.arange(len(signal)) / sample_rate
    ax.fill_between(time, signal, color=COLORS["primary"], alpha=0.18, linewidth=0)
    ax.plot(time, signal, linewidth=1.2, color=COLORS["primary"])
    _add_zero_line(ax)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, time[-1]])

    return _finalize_figure(fig, save_path)


def plot_spectrogram(signal: np.ndarray, sample_rate: int,
                     n_fft: int = 512, frame_len_ms: float = 25,
                     frame_shift_ms: float = 10,
                     title: str = "Spectrogram",
                     save_path: str = None, figsize: tuple = (12, 4)):
    """
    Plot spectrogram (time-frequency representation).

    Args:
        signal: Audio signal
        sample_rate: Sample rate in Hz
        n_fft: FFT size
        frame_len_ms: Frame length in ms
        frame_shift_ms: Frame shift in ms
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to tensor for processing
    if isinstance(signal, np.ndarray):
        signal_tensor = torch.from_numpy(signal).float()
    else:
        signal_tensor = signal.float()

    # Compute STFT
    frame_len = int(sample_rate * frame_len_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)

    signal_tensor = pre_emphasis(signal_tensor)
    frames = frame_signal(signal_tensor, frame_len, frame_shift)
    frames = apply_window(frames, 'hamming')
    power_spec = compute_power_spectrum(frames, n_fft)

    # Convert to dB
    spec_db = 10 * np.log10(power_spec.numpy() + 1e-10)
    floor, ceil = np.percentile(spec_db, [1, 99])
    spec_db = np.clip(spec_db, floor, ceil)

    # Plot
    time_axis = np.arange(spec_db.shape[0]) * frame_shift_ms / 1000
    freq_axis = np.arange(spec_db.shape[1]) * sample_rate / n_fft

    im = ax.pcolormesh(time_axis, freq_axis, spec_db.T,
                       shading='gouraud', cmap='magma', rasterized=True)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, sample_rate / 2])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)', fontsize=10)

    return _finalize_figure(fig, save_path)


def plot_mfcc_heatmap(mfcc: np.ndarray, sample_rate: int = None,
                     frame_shift_ms: float = 10,
                     title: str = "MFCC Features",
                     save_path: str = None, figsize: tuple = (12, 5)):
    """
    Plot MFCC coefficients as heatmap.

    Args:
        mfcc: MFCC features [num_frames, n_mfcc]
        sample_rate: Sample rate (optional, for time axis)
        frame_shift_ms: Frame shift in ms
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert tensor to numpy if needed
    if torch.is_tensor(mfcc):
        mfcc = mfcc.cpu().numpy()

    display_mfcc, vlim = _zscore_for_display(mfcc)

    # Time axis
    time_axis = np.arange(mfcc.shape[0]) * frame_shift_ms / 1000
    xticks = np.linspace(0, mfcc.shape[0] - 1, 6, dtype=int)
    xtick_labels = [f"{time_axis[idx]:.2f}" for idx in xticks]

    # Plot heatmap with softer palette
    im = sns.heatmap(
        display_mfcc.T,
        ax=ax,
        cmap=MFCC_CMAP,
        cbar_kws={'label': 'Coefficient Value'},
        xticklabels=xtick_labels,
        yticklabels=[f'C{i}' for i in range(mfcc.shape[1])],
        vmin=-vlim,
        vmax=vlim,
        center=0
    )

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('MFCC Coefficient', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(xticks + 0.5)

    return _finalize_figure(fig, save_path)


def plot_mel_filterbank(n_filters: int = 26, n_fft: int = 512,
                        sample_rate: int = 44100,
                        title: str = "Mel Filterbank",
                        save_path: str = None, figsize: tuple = (12, 5)):
    """
    Plot Mel filterbank frequency response.

    Args:
        n_filters: Number of Mel filters
        n_fft: FFT size
        sample_rate: Sample rate in Hz
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create filterbank
    fbank = create_mel_filterbank(n_filters, n_fft, sample_rate, device='cpu')
    fbank = fbank.numpy()

    # Frequency axis
    freq_axis = np.arange(fbank.shape[1]) * sample_rate / n_fft

    # Plot each filter
    colors = sns.color_palette("crest", n_filters)
    for i in range(n_filters):
        ax.plot(freq_axis, fbank[i], color=colors[i], alpha=0.85, linewidth=1.2)
        ax.fill_between(freq_axis, fbank[i], color=colors[i], alpha=0.08)

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, sample_rate / 2])
    ax.set_ylim([0, 1.05])

    return _finalize_figure(fig, save_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: list = None,
                          title: str = "Confusion Matrix",
                          save_path: str = None, figsize: tuple = (10, 8)):
    """
    Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix [n_classes, n_classes]
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]

    # Normalize for display (show percentages)
    row_sums = cm.sum(axis=1, keepdims=True) + 1e-12
    cm_normalized = cm.astype('float') / row_sums

    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Proportion'})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    return _finalize_figure(fig, save_path)


def compute_dtw_alignment(seq1: np.ndarray, seq2: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """Compute local distance, cumulative cost, and optimal path for DTW."""
    if torch.is_tensor(seq1):
        seq1 = seq1.cpu().numpy()
    if torch.is_tensor(seq2):
        seq2 = seq2.cpu().numpy()

    n, m = seq1.shape[0], seq2.shape[0]
    dist_matrix = np.sqrt(((seq1[:, np.newaxis, :] - seq2[np.newaxis, :, :]) ** 2).sum(axis=2))

    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            step_cost = dist_matrix[i - 1, j - 1]
            cost[i, j] = step_cost + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Backtrace optimal path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        neighbors = [cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]]
        step = int(np.argmin(neighbors))
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()

    return dist_matrix, cost[1:, 1:], path


def plot_dtw_matrix(seq1: np.ndarray, seq2: np.ndarray,
                    title: str = "DTW Distance Matrix",
                    save_path: str = None, figsize: tuple = (10, 8)):
    """
    Plot DTW cost matrix with optimal path.

    Args:
        seq1: Test sequence [N, feature_dim]
        seq2: Reference template [M, feature_dim]
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    dist_matrix, cost_matrix, path = compute_dtw_alignment(seq1, seq2)

    im = sns.heatmap(cost_matrix, ax=ax, cmap="rocket_r",
                     cbar_kws={'label': 'Cumulative cost'}, linewidths=0,
                     xticklabels=False, yticklabels=False)

    if path:
        xs = [j + 0.5 for _, j in path]
        ys = [i + 0.5 for i, _ in path]
        ax.plot(xs, ys, color=COLORS["accent"], linewidth=2.2, alpha=0.9, label='Optimal path')
        ax.scatter(xs[0], ys[0], color=COLORS["secondary"], s=48, zorder=5, marker='o', label='Start')
        ax.scatter(xs[-1], ys[-1], color=COLORS["primary"], s=60, zorder=5, marker='X', label='End')
        legend = ax.legend(
            loc='upper right',
            frameon=True,
            fontsize=9,
            facecolor='white',
            edgecolor='#e5e7eb',
            framealpha=0.9
        )

    ax.set_xlabel('Reference Frame', fontsize=12)
    ax.set_ylabel('Test Frame', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    return _finalize_figure(fig, save_path)


def plot_template_comparison(templates: dict, digit_list: list = None,
                             title: str = "MFCC Template Comparison",
                             save_path: str = None, figsize: tuple = (15, 10)):
    """
    Plot MFCC templates for multiple digits.

    Args:
        templates: Dictionary {digit: mfcc_tensor}
        digit_list: List of digits to plot (default: all)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if digit_list is None:
        digit_list = sorted(templates.keys(), key=lambda x: int(x))

    template_arrays = []
    display_arrays = []
    for d in digit_list:
        if d in templates:
            arr = templates[d]
            if torch.is_tensor(arr):
                arr = arr.cpu().numpy()
            template_arrays.append(arr)
            norm_arr, _ = _zscore_for_display(arr)
            display_arrays.append(norm_arr)

    if display_arrays:
        stacked = np.concatenate(display_arrays)
        vlim = np.percentile(np.abs(stacked), 97)
    else:
        vlim = None

    n_digits = len(digit_list)
    n_cols = 5
    n_rows = (n_digits + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_digits > 1 else [axes]

    for i, digit in enumerate(digit_list):
        if digit not in templates:
            continue

        mfcc = templates[digit]
        if torch.is_tensor(mfcc):
            mfcc = mfcc.cpu().numpy()
        display_mfcc, _ = _zscore_for_display(mfcc)

        ax = axes[i]
        sns.heatmap(
            display_mfcc.T,
            ax=ax,
            cmap=MFCC_CMAP,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            vmin=-vlim if vlim is not None else None,
            vmax=vlim if vlim is not None else None,
            center=0
        )

        ax.set_title(f'Digit {digit}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Frame', fontsize=9)
        ax.set_ylabel('MFCC', fontsize=9)

    # Hide empty subplots
    for i in range(n_digits, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.subplots_adjust(top=0.92)
    return _finalize_figure(fig, save_path)


def plot_recognition_distances(distances: dict, true_label: str = None,
                               title: str = "DTW Distances to Templates",
                               save_path: str = None, figsize: tuple = (10, 6)):
    """
    Plot DTW distances to all templates (bar chart).

    Args:
        distances: Dictionary {digit: dtw_distance}
        true_label: True label for highlighting
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    digits = sorted(distances.keys(), key=lambda x: int(x))
    dists = [distances[d] for d in digits]

    # Colors: highlight predicted (min) and true
    colors = [COLORS["primary"]] * len(digits)
    min_idx = np.argmin(dists)
    colors[min_idx] = COLORS["accent"]  # Predicted

    if true_label and true_label in digits:
        true_idx = digits.index(true_label)
        if true_idx != min_idx:
            colors[true_idx] = COLORS["secondary"]  # True but wrong

    bars = ax.bar(digits, dists, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Digit', fontsize=12)
    ax.set_ylabel('DTW Distance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, dist in zip(bars, dists):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{dist:.1f}', ha='center', va='bottom', fontsize=9)

    return _finalize_figure(fig, save_path)


def plot_accuracy_comparison(results: dict,
                             title: str = "Recognition Accuracy by Digit",
                             save_path: str = None, figsize: tuple = (10, 6)):
    """
    Plot per-digit recognition accuracy.

    Args:
        results: Dictionary from confusion_matrix_report
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    report = results['classification_report']

    # Extract per-class metrics
    digits = []
    precisions = []
    recalls = []
    f1_scores = []

    for key, metrics in report.items():
        if key.isdigit():
            digits.append(key)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1-score'])

    x = np.arange(len(digits))
    width = 0.25

    ax.bar(x - width, precisions, width, label='Precision', color=COLORS["primary"])
    ax.bar(x, recalls, width, label='Recall', color=COLORS["accent"])
    ax.bar(x + width, f1_scores, width, label='F1-Score', color=COLORS["secondary"])

    ax.set_xlabel('Digit', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(digits)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.1])

    # Add overall accuracy line
    overall_acc = results['accuracy']
    ax.axhline(y=overall_acc, color='#4caf50', linestyle='--',
               label=f'Overall Acc: {overall_acc:.2%}', alpha=0.8)
    ax.legend(loc='lower right')

    return _finalize_figure(fig, save_path)


def generate_all_visualizations(signal: np.ndarray, sample_rate: int,
                                templates: dict, results: dict,
                                output_dir: str, digit_label: str = '0'):
    """
    Generate all visualizations for experiment report.

    Args:
        signal: Sample audio signal
        sample_rate: Sample rate
        templates: Trained templates
        results: Evaluation results
        output_dir: Output directory for figures
        digit_label: Label of the sample signal
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating visualizations to: {output_dir}")

    # 1. Waveform
    plot_waveform(signal, sample_rate,
                  title=f"Audio Waveform - Digit {digit_label}",
                  save_path=os.path.join(output_dir, 'waveform.png'))

    # 2. Spectrogram
    plot_spectrogram(signal, sample_rate,
                     title=f"Spectrogram - Digit {digit_label}",
                     save_path=os.path.join(output_dir, 'spectrogram.png'))

    # 3. MFCC
    mfcc = extract_mfcc(signal, sample_rate)
    plot_mfcc_heatmap(mfcc,
                      title=f"MFCC Features - Digit {digit_label}",
                      save_path=os.path.join(output_dir, 'mfcc_heatmap.png'))

    # 4. Mel Filterbank
    plot_mel_filterbank(sample_rate=sample_rate,
                        title="Mel Filterbank Frequency Response",
                        save_path=os.path.join(output_dir, 'mel_filterbank.png'))

    # 5. Template Comparison
    plot_template_comparison(templates,
                             title="MFCC Templates for All Digits",
                             save_path=os.path.join(output_dir, 'template_comparison.png'))

    # 6. Confusion Matrix
    plot_confusion_matrix(results['confusion_matrix'],
                          class_names=[str(i) for i in range(10)],
                          title="Recognition Confusion Matrix",
                          save_path=os.path.join(output_dir, 'confusion_matrix.png'))

    # 7. Per-digit Accuracy
    plot_accuracy_comparison(results,
                             title="Per-Digit Recognition Performance",
                             save_path=os.path.join(output_dir, 'accuracy_comparison.png'))

    # 8. Distances to all templates for the sample
    try:
        from .recognizer import recognize_single

        recognized_digit, distances = recognize_single(
            signal, sample_rate, templates,
            n_mfcc=mfcc.shape[1], frame_len_ms=25, frame_shift_ms=10,
            verbose=False
        )
        print(f"DTW sample digit={digit_label}, recognized={recognized_digit}")

        plot_recognition_distances(
            distances,
            true_label=digit_label,
            title=f"DTW Distances (True {digit_label}, Pred {recognized_digit})",
            save_path=os.path.join(output_dir, 'dtw_distances.png')
        )

        target_digit = digit_label if digit_label in templates else recognized_digit
        template_mfcc = templates.get(target_digit)
        if template_mfcc is not None:
            plot_dtw_matrix(
                mfcc,
                template_mfcc,
                title=f"DTW Alignment Path (Sample {digit_label} vs Template {target_digit})",
                save_path=os.path.join(output_dir, 'dtw_alignment.png')
            )
    except Exception as e:
        print(f"Warning: Skipped DTW alignment plot due to error: {e}")

    print(f"Generated visualization figures in {output_dir}.")
