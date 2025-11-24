"""
MFCC template trainer for DTW-based recognition.
"""
import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .features import extract_mfcc


def split_dataset(data_dir: str, test_size: float = 0.2,
                  random_state: int = 42) -> tuple[dict, dict]:
    """
    Split dataset into train and test sets.

    Args:
        data_dir: Root directory containing digit folders (0-9)
        test_size: Proportion of test set (default 0.2)
        random_state: Random seed for reproducibility

    Returns:
        train_files: {digit: [file_paths]}
        test_files: {digit: [file_paths]}
    """
    train_files = {}
    test_files = {}

    for digit in range(10):
        digit_str = str(digit)
        digit_dir = os.path.join(data_dir, digit_str)

        if not os.path.exists(digit_dir):
            continue

        wav_files = glob(os.path.join(digit_dir, '*.wav'))
        if len(wav_files) < 2:
            # Not enough samples to split
            train_files[digit_str] = wav_files
            test_files[digit_str] = []
            continue

        # Split files
        train, test = train_test_split(
            wav_files,
            test_size=test_size,
            random_state=random_state
        )
        train_files[digit_str] = train
        test_files[digit_str] = test

    return train_files, test_files


def train_templates_from_files(train_files: dict, sample_rate: int = 44100,
                               n_mfcc: int = 13, n_fft: int = 512,
                               frame_len_ms: float = 25, frame_shift_ms: float = 10,
                               template_strategy: str = 'first',
                               device: str = None) -> dict:
    """
    Train MFCC templates from file lists.

    Args:
        train_files: Dictionary {digit: [file_paths]}
        sample_rate: Expected sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT size
        frame_len_ms: Frame length in ms
        frame_shift_ms: Frame shift in ms
        template_strategy: 'first' (use first sample) or 'mean' (average all)
        device: Computation device

    Returns:
        Dictionary of templates {digit_str: mfcc_tensor}
    """
    from src.common.audio_io import load_audio

    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    templates = {}

    print(f"Device: {device}")

    # Process each digit
    for digit_str, wav_files in train_files.items():
        if not wav_files:
            print(f"Warning: No files for digit {digit_str}")
            continue

        print(f"Processing digit {digit_str}: {len(wav_files)} train samples")

        # Extract MFCC for all samples
        all_mfccs = []
        for wav_file in wav_files:
            try:
                signal, sr = load_audio(wav_file)

                # Resample if needed (simple warning for now)
                if sr != sample_rate:
                    print(f"  Warning: {wav_file} has sample rate {sr}, expected {sample_rate}")

                mfcc = extract_mfcc(
                    signal, sr,
                    n_mfcc=n_mfcc, n_fft=n_fft,
                    frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
                    device=device
                )
                all_mfccs.append(mfcc.cpu())

            except Exception as e:
                print(f"  Error processing {wav_file}: {e}")

        if not all_mfccs:
            continue

        # Select template based on strategy
        if template_strategy == 'first':
            template = all_mfccs[0]
        elif template_strategy == 'mean':
            # Average MFCCs (need to handle variable lengths)
            # Simple approach: use median length and truncate/pad
            lengths = [m.shape[0] for m in all_mfccs]
            median_len = int(np.median(lengths))

            padded = []
            for m in all_mfccs:
                if m.shape[0] >= median_len:
                    padded.append(m[:median_len])
                else:
                    pad = torch.zeros(median_len - m.shape[0], m.shape[1])
                    padded.append(torch.cat([m, pad]))

            template = torch.stack(padded).mean(dim=0)
        else:
            template = all_mfccs[0]

        templates[digit_str] = template
        print(f"  Template shape: {template.shape}")

    return templates


def save_templates(templates: dict, save_path: str):
    """
    Save templates to file.

    Args:
        templates: Dictionary of templates
        save_path: Output file path (.pt)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(templates, save_path)
    print(f"Templates saved to: {save_path}")


def load_templates(load_path: str) -> dict:
    """
    Load templates from file.

    Args:
        load_path: Template file path (.pt)

    Returns:
        Dictionary of templates
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Template file not found: {load_path}")

    templates = torch.load(load_path, map_location='cpu')
    print(f"Templates loaded from: {load_path}")
    print(f"Available digits: {list(templates.keys())}")

    return templates
