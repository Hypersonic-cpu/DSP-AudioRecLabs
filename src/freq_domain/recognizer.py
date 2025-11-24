"""
DTW-based speech recognizer for digit recognition.
"""
import torch
import numpy as np
from glob import glob
import os
from tqdm import tqdm

from .features import extract_mfcc
import config
from src.time_domain.audio_processing import endpoint_detection
import numpy as np
from .dtw import dtw_distance


def recognize_single(signal: np.ndarray, sample_rate: int, templates: dict,
                     n_mfcc: int = 13, n_fft: int = 512,
                     frame_len_ms: float = 25, frame_shift_ms: float = 10,
                     device: str = None, verbose: bool = False) -> tuple[str, dict]:
    """
    Recognize a single audio signal using DTW template matching.

    Args:
        signal: Audio signal (numpy array)
        sample_rate: Sample rate
        templates: Dictionary of templates {digit: mfcc_tensor}
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT size
        frame_len_ms: Frame length in ms
        frame_shift_ms: Frame shift in ms
        device: Computation device
        verbose: Print distances

    Returns:
        recognized_digit: The recognized digit as string
        distances: Dictionary of DTW distances to each template
    """
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Endpoint detection to get ROI and normalize by average energy
    try:
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

    # Extract MFCC from input ROI
    input_mfcc = extract_mfcc(
        signal_roi, sample_rate,
        n_mfcc=n_mfcc, n_fft=n_fft,
        frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
        device=device
    )

    # Compute DTW distance to each template
    distances = {}
    for digit, template in templates.items():
        if not torch.is_tensor(template):
            template = torch.from_numpy(template).float()
        template = template.to(device)

        dist = dtw_distance(input_mfcc, template)
        distances[digit] = dist

        if verbose:
            print(f"  Distance to '{digit}': {dist:.2f}")

    # Find minimum distance
    recognized = min(distances, key=distances.get)

    return recognized, distances


def recognize_file(filepath: str, templates: dict, **kwargs) -> tuple[str, dict]:
    """
    Recognize digit from audio file.

    Args:
        filepath: Path to audio file
        templates: Dictionary of templates
        **kwargs: Additional arguments for recognize_single

    Returns:
        recognized_digit, distances
    """
    from src.common.audio_io import load_audio

    signal, sample_rate = load_audio(filepath)
    return recognize_single(signal, sample_rate, templates, **kwargs)


def evaluate_dataset(data_dir: str, templates: dict,
                     n_mfcc: int = 13, n_fft: int = 512,
                     frame_len_ms: float = 25, frame_shift_ms: float = 10,
                     device: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate recognition on entire dataset.

    Args:
        data_dir: Directory containing digit folders (0-9)
        templates: Dictionary of templates
        n_mfcc, n_fft, frame_len_ms, frame_shift_ms: MFCC parameters
        device: Computation device

    Returns:
        y_true: Ground truth labels
        y_pred: Predicted labels
    """
    from src.common.audio_io import load_audio

    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    y_true = []
    y_pred = []

    print(f"Evaluating on: {data_dir}")

    # Process each digit
    for digit in range(10):
        digit_str = str(digit)
        digit_dir = os.path.join(data_dir, digit_str)

        if not os.path.exists(digit_dir):
            continue

        wav_files = glob(os.path.join(digit_dir, '*.wav'))
        print(f"Digit {digit}: {len(wav_files)} samples")

        for wav_file in tqdm(wav_files, desc=f"Digit {digit}", leave=False):
            try:
                signal, sr = load_audio(wav_file)

                recognized, _ = recognize_single(
                    signal, sr, templates,
                    n_mfcc=n_mfcc, n_fft=n_fft,
                    frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
                    device=device, verbose=False
                )

                y_true.append(digit)
                y_pred.append(int(recognized))

            except Exception as e:
                print(f"Error processing {wav_file}: {e}")

    return np.array(y_true), np.array(y_pred)


def evaluate_files(test_files: dict, templates: dict,
                   n_mfcc: int = 13, n_fft: int = 512,
                   frame_len_ms: float = 25, frame_shift_ms: float = 10,
                   device: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate recognition on test file lists.

    Args:
        test_files: Dictionary {digit: [file_paths]}
        templates: Dictionary of templates
        n_mfcc, n_fft, frame_len_ms, frame_shift_ms: MFCC parameters
        device: Computation device

    Returns:
        y_true: Ground truth labels
        y_pred: Predicted labels
    """
    from src.common.audio_io import load_audio

    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    y_true = []
    y_pred = []

    total_files = sum(len(files) for files in test_files.values())
    print(f"Evaluating on {total_files} test samples")

    # Process each digit
    for digit_str, wav_files in test_files.items():
        digit = int(digit_str)
        print(f"Digit {digit}: {len(wav_files)} test samples")

        for wav_file in tqdm(wav_files, desc=f"Digit {digit}", leave=False):
            try:
                signal, sr = load_audio(wav_file)

                recognized, _ = recognize_single(
                    signal, sr, templates,
                    n_mfcc=n_mfcc, n_fft=n_fft,
                    frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
                    device=device, verbose=False
                )

                y_true.append(digit)
                y_pred.append(int(recognized))

            except Exception as e:
                print(f"Error processing {wav_file}: {e}")

    return np.array(y_true), np.array(y_pred)
