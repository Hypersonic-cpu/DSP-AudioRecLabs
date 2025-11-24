"""
Frequency domain speech recognition module.
Implements MFCC feature extraction and DTW-based recognition.
"""
from .features import (
    extract_mfcc, extract_mfcc_with_delta,
    extract_gfcc, extract_gfcc_with_delta,
    compute_delta, compute_delta_delta,
    create_mel_filterbank, create_gammatone_filterbank
)
from .dtw import dtw_distance, dtw_distance_fast
from .trainer import (
    split_dataset, train_templates_from_files,
    save_templates, load_templates
)
from .recognizer import recognize_single, recognize_file, evaluate_dataset, evaluate_files
from .visualization import (
    plot_waveform, plot_spectrogram, plot_mfcc_heatmap,
    plot_mel_filterbank, plot_confusion_matrix, plot_dtw_matrix,
    plot_template_comparison, plot_recognition_distances,
    plot_accuracy_comparison, generate_all_visualizations
)

__all__ = [
    # Feature extraction
    'extract_mfcc',
    'extract_mfcc_with_delta',
    'extract_gfcc',
    'extract_gfcc_with_delta',
    'compute_delta',
    'compute_delta_delta',
    'create_mel_filterbank',
    'create_gammatone_filterbank',
    # DTW
    'dtw_distance',
    'dtw_distance_fast',
    'split_dataset',
    'train_templates_from_files',
    'save_templates',
    'load_templates',
    'recognize_single',
    'recognize_file',
    'evaluate_dataset',
    'evaluate_files',
    # Visualization
    'plot_waveform',
    'plot_spectrogram',
    'plot_mfcc_heatmap',
    'plot_mel_filterbank',
    'plot_confusion_matrix',
    'plot_dtw_matrix',
    'plot_template_comparison',
    'plot_recognition_distances',
    'plot_accuracy_comparison',
    'generate_all_visualizations'
]
