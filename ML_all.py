#!/usr/bin/env python3
"""
Train and infer an MLP using combined time-domain statistical features
and frequency-domain MFCC (with optional delta) statistical features.

Usage:
  Training:
    export SPEECH_DATA_DIR=./data
    python ML_all.py --mode train --results-dir ./results --n-mfcc 13

  Inference:
    python ML_all.py --mode infer --model ./results/ml_all_checkpoint.pt --file test.wav

This script reuses existing processing functions from `src.time_domain` and
`src.freq_domain` to ensure consistent endpoint detection, framing and
feature extraction.
"""
import os
import sys
import argparse
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.time_domain.audio_processing import (
    process_audio_file, preprocess
)
from src.time_domain.feature_extraction import (
    extract_features_from_frames, normalize_features, compute_statistics
)
from src.driver.loader import load_wav
from src.freq_domain.features import extract_mfcc_with_delta
from src.time_domain.models import create_classifier, MLPTrainer
import torch


def find_data_dir(provided_dir=None):
    if provided_dir:
        return os.path.abspath(os.path.expanduser(provided_dir))
    env = os.getenv('SPEECH_DATA_DIR')
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.abspath(os.path.expanduser(config.DATA_DIR))


def mfcc_stats_vector(mfcc_tensor):
    """Convert MFCC frames (torch.Tensor or np.ndarray) to statistical vector.

    For each MFCC coefficient (column) compute mean,std,max,min,median.
    """
    # Accept torch tensor or numpy
    try:
        import torch
        if 'torch' in str(type(mfcc_tensor)):
            mfcc = mfcc_tensor.cpu().numpy()
        else:
            mfcc = np.array(mfcc_tensor)
    except Exception:
        mfcc = np.array(mfcc_tensor)

    if mfcc is None or mfcc.size == 0:
        return np.zeros(0), []

    # mfcc: (n_frames, n_coeffs)
    if mfcc.ndim == 1:
        mfcc = mfcc.reshape(-1, 1)

    n_coeffs = mfcc.shape[1]
    vec = []
    names = []
    for i in range(n_coeffs):
        col = mfcc[:, i]
        stats = compute_statistics(col)
        for stat_name, stat_value in stats.items():
            vec.append(stat_value)
            names.append(f'mfcc_{i}_{stat_name}')

    return np.array(vec), names


def load_dataset_all(data_dir, n_mfcc=13, include_delta=True, include_delta_delta=True,
                     window_type='hamming', do_endpoint_detection=True):
    class_names = [str(i) for i in range(10)]

    X = []
    y = []
    feature_names = None

    for class_idx, cls in enumerate(class_names):
        class_path = os.path.join(data_dir, cls)
        if not os.path.isdir(class_path):
            print(f"Warning: class folder not found, skipping: {class_path}")
            continue

        wav_files = glob(os.path.join(class_path, '*.wav'))
        print(f"Found {len(wav_files)} files for class {cls}")

        for wav_file in wav_files:
            try:
                # time-domain frames + metadata
                frames, sr, meta = process_audio_file(
                    wav_file,
                    frame_length=config.FRAME_LENGTH,
                    frame_shift=config.FRAME_SHIFT,
                    window_type=window_type,
                    do_endpoint_detection=do_endpoint_detection,
                    energy_high_ratio=config.ENERGY_HIGH_RATIO,
                    energy_low_ratio=config.ENERGY_LOW_RATIO,
                    zcr_threshold_ratio=config.ZCR_THRESHOLD_RATIO
                )

                # time-domain statistical features
                time_vec, time_names = extract_features_from_frames(frames, method='statistical')

                # load raw audio and slice ROI for MFCC
                signal, sr2 = load_wav(wav_file)
                signal = preprocess(signal)

                # determine ROI from metadata if available
                start = meta.get('start_point', 0)
                end = meta.get('end_point', len(signal))
                if start is None:
                    start = 0
                if end is None or end <= start:
                    end = len(signal)

                signal_roi = signal[start:end] if len(signal) > 0 else signal

                # energy normalization of ROI
                try:
                    mean_energy = np.mean(np.array(signal_roi) ** 2)
                    if mean_energy > 0:
                        signal_roi = signal_roi / float(np.sqrt(mean_energy))
                except Exception:
                    pass

                # frequency-domain features (MFCC + deltas)
                try:
                    mfcc_feat = extract_mfcc_with_delta(
                        signal_roi, sr2,
                        n_mfcc=n_mfcc,
                        n_fft=config.N_FFT if hasattr(config, 'N_FFT') else 512,
                        n_filters=26,
                        frame_len_ms=config.FRAME_LENGTH_MS if hasattr(config, 'FRAME_LENGTH_MS') else 25,
                        frame_shift_ms=config.FRAME_SHIFT_MS if hasattr(config, 'FRAME_SHIFT_MS') else 10,
                        include_delta=include_delta,
                        include_delta_delta=include_delta_delta
                    )
                except Exception as e:
                    # fallback: try simple MFCC
                    print(f"MFCC extraction failed for {wav_file}: {e}")
                    mfcc_feat = np.zeros((1, n_mfcc))

                mfcc_vec, mfcc_names = mfcc_stats_vector(mfcc_feat)

                # combine
                if time_vec.size == 0 and mfcc_vec.size == 0:
                    continue

                combined = np.concatenate([time_vec, mfcc_vec])

                if feature_names is None:
                    feature_names = list(time_names) + list(mfcc_names)

                X.append(combined)
                y.append(class_idx)

            except Exception as e:
                print(f"  Error processing {wav_file}: {e}")
                continue

    if len(X) == 0:
        raise RuntimeError(f"No samples found in data_dir: {data_dir}")

    X = np.array(X)
    y = np.array(y)

    return X, y, feature_names, class_names


def train_mode(args):
    data_dir = find_data_dir(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(2)

    results_dir = os.path.abspath(os.path.expanduser(args.results_dir or './results'))
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading dataset from: {data_dir}")
    X, y, feat_names, class_names = load_dataset_all(
        data_dir,
        n_mfcc=args.n_mfcc,
        include_delta=not args.no_delta,
        include_delta_delta=not args.no_delta_delta,
        window_type=args.window_type
    )

    print(f"Total samples: {len(X)}, feature dim: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    X_train_norm, mean, std = normalize_features(X_train)
    X_test_norm, _, _ = normalize_features(X_test, mean, std)

    input_size = X_train_norm.shape[1]
    hidden_layers = args.hidden_layers if args.hidden_layers is not None else config.MLP_HIDDEN_LAYERS
    num_classes = len(class_names)
    epochs = args.epochs if args.epochs is not None else config.MLP_EPOCHS
    batch_size = args.batch_size if args.batch_size is not None else config.MLP_BATCH_SIZE

    print(f"Creating MLP: input_size={input_size}, hidden={hidden_layers}, classes={num_classes}")

    trainer = create_classifier(
        'mlp',
        input_size=input_size,
        hidden_layers=hidden_layers,
        num_classes=num_classes,
        learning_rate=config.MLP_LEARNING_RATE,
        epochs=epochs,
        batch_size=batch_size
    )

    checkpoint_path = os.path.join(results_dir, args.save_name)

    print("Start training...")
    trainer.fit(X_train_norm, y_train, verbose=True, save_path=checkpoint_path, mean=mean, std=std, class_names=class_names)

    results = trainer.evaluate(X_test_norm, y_test)
    print(f"Test accuracy: {results['accuracy']:.4f}")

    summary_path = os.path.join(results_dir, 'ml_all_training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('ML All Training Summary\n')
        f.write('====================\n')
        f.write(f'Train samples: {len(X_train)}\n')
        f.write(f'Test samples: {len(X_test)}\n')
        f.write(f'Feature dim: {input_size}\n')
        f.write(f'Hidden layers: {hidden_layers}\n')
        f.write(f'Epochs: {epochs}\n')
        f.write(f'Batch size: {batch_size}\n')
        f.write(f'Test accuracy: {results["accuracy"]:.6f}\n')

    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Training summary saved to: {summary_path}")


def infer_mode(args):
    if not args.model:
        print("Error: please specify --model for inference")
        return
    if not args.file:
        print("Error: please specify --file (wav) to recognize")
        return

    if not os.path.exists(args.model):
        print(f"Error: model not found: {args.model}")
        return

    trainer = MLPTrainer.load_from_checkpoint(args.model)

    # build feature vector for the file (same pipeline as training)
    wav_file = args.file
    try:
        frames, sr, meta = process_audio_file(
            wav_file,
            frame_length=config.FRAME_LENGTH,
            frame_shift=config.FRAME_SHIFT,
            window_type=args.window_type,
            do_endpoint_detection=True,
            energy_high_ratio=config.ENERGY_HIGH_RATIO,
            energy_low_ratio=config.ENERGY_LOW_RATIO,
            zcr_threshold_ratio=config.ZCR_THRESHOLD_RATIO
        )

        time_vec, _ = extract_features_from_frames(frames, method='statistical')

        signal, sr2 = load_wav(wav_file)
        signal = preprocess(signal)
        start = meta.get('start_point', 0)
        end = meta.get('end_point', len(signal))
        if start is None:
            start = 0
        if end is None or end <= start:
            end = len(signal)
        signal_roi = signal[start:end]

        try:
            mean_energy = np.mean(np.array(signal_roi) ** 2)
            if mean_energy > 0:
                signal_roi = signal_roi / float(np.sqrt(mean_energy))
        except Exception:
            pass

        mfcc_feat = extract_mfcc_with_delta(
            signal_roi, sr2,
            n_mfcc=args.n_mfcc,
            n_fft=config.N_FFT if hasattr(config, 'N_FFT') else 512,
            n_filters=26,
            frame_len_ms=config.FRAME_LENGTH_MS if hasattr(config, 'FRAME_LENGTH_MS') else 25,
            frame_shift_ms=config.FRAME_SHIFT_MS if hasattr(config, 'FRAME_SHIFT_MS') else 10,
            include_delta=not args.no_delta,
            include_delta_delta=not args.no_delta_delta
        )

        mfcc_vec, _ = mfcc_stats_vector(mfcc_feat)

        combined = np.concatenate([time_vec, mfcc_vec])

        # normalize using trainer.mean/std
        mean = trainer.mean
        std = trainer.std
        if mean is None or std is None:
            print("Warning: model does not contain normalization stats; proceeding without normalization")
            X_norm = combined.reshape(1, -1)
        else:
            # Ensure mean/std are numpy arrays
            mean = np.array(mean)
            std = np.array(std)

            # If shapes mismatch, try to align MFCC part to training dimensions
            if combined.shape[0] != mean.shape[0]:
                # assume time_vec length equals the first part of mean
                t_len = time_vec.shape[0]
                if t_len <= mean.shape[0]:
                    mfcc_train_len = mean.shape[0] - t_len

                    # extract current mfcc vec
                    cur_mfcc = mfcc_vec
                    # adjust cur_mfcc to mfcc_train_len
                    if cur_mfcc.shape[0] < mfcc_train_len:
                        # pad with zeros
                        pad = np.zeros(mfcc_train_len - cur_mfcc.shape[0], dtype=cur_mfcc.dtype)
                        cur_mfcc_adj = np.concatenate([cur_mfcc, pad])
                    else:
                        # truncate
                        cur_mfcc_adj = cur_mfcc[:mfcc_train_len]

                    combined = np.concatenate([time_vec, cur_mfcc_adj])
                else:
                    # fallback: if time_vec longer than mean, truncate time_vec
                    combined = combined[:mean.shape[0]]

            # final check: if still mismatch, pad or truncate to mean length
            if combined.shape[0] != mean.shape[0]:
                target = mean.shape[0]
                if combined.shape[0] < target:
                    pad = np.zeros(target - combined.shape[0], dtype=combined.dtype)
                    combined = np.concatenate([combined, pad])
                else:
                    combined = combined[:target]

            # avoid division by zero in std
            std_safe = np.where(std == 0, 1.0, std)
            X_norm = ((combined - mean) / std_safe).reshape(1, -1)

        # Get model logits and softmax confidences
        trainer.model.eval()
        with torch.no_grad():
            device = trainer.device if hasattr(trainer, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
            input_tensor = torch.FloatTensor(X_norm).to(device)
            logits = trainer.model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

        pred = int(np.argmax(probs))
        label = None
        if trainer.class_names:
            try:
                label = trainer.class_names[int(pred)]
            except Exception:
                label = str(pred)
        else:
            label = str(pred)

        # Print confidences for all classes
        print("\nClass confidences (softmax probabilities):")
        for idx, p in enumerate(probs):
            name = trainer.class_names[idx] if trainer.class_names else str(idx)
            print(f"  {name}: {p:.4f}")

        print(f"\nRecognized digit: {label}")

    except Exception as e:
        print(f"Error during inference: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train/infer combined time+freq MLP')
    parser.add_argument('--mode', choices=['train', 'infer'], required=True)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--results-dir', default='./results')
    parser.add_argument('--save-name', default='ml_all_checkpoint.pt')
    parser.add_argument('--model', default=None)
    parser.add_argument('--file', default=None)
    parser.add_argument('--n-mfcc', type=int, default=13)
    parser.add_argument('--no-delta', action='store_true', help='Do not include delta features')
    parser.add_argument('--no-delta-delta', action='store_true', help='Do not include delta-delta features')
    parser.add_argument('--window-type', default='hamming')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--hidden-layers', nargs='*', type=int, default=None)

    args = parser.parse_args()

    if args.mode == 'train':
        train_mode(args)
    else:
        infer_mode(args)


if __name__ == '__main__':
    main()
