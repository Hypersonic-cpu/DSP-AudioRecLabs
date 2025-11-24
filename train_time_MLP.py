#!/usr/bin/env python3
"""
Train MLP on time-domain statistical features (one-shot for digits 0-9).

This script will:
 - locate dataset directory from environment variable `SPEECH_DATA_DIR` or command-line/config
 - expect subfolders named `0`..`9` each containing .wav files
 - process audio (endpoint detection, framing), extract statistical features, normalize
 - split into train/test, train the MLP using `MLPTrainer` from `src/time_domain/models.py`
 - save a checkpoint containing model_state_dict, architecture info, mean/std, and class_names

Usage:
  export SPEECH_DATA_DIR=./data
  python train_time_MLP.py --results-dir ./results --epochs 50

"""
import os
import sys
import argparse
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split

# allow imports relative to repo root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.time_domain.audio_processing import process_audio_file, load_wav, preprocess
from src.time_domain.feature_extraction import (
    extract_features_from_frames, normalize_features
)
from src.time_domain.models import create_classifier


def find_data_dir(provided_dir=None):
    # priority: provided_dir -> env SPEECH_DATA_DIR -> config.DATA_DIR
    if provided_dir:
        return os.path.abspath(os.path.expanduser(provided_dir))
    env = os.getenv('SPEECH_DATA_DIR')
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.abspath(os.path.expanduser(config.DATA_DIR))


def load_dataset(data_dir, window_type='hamming', do_endpoint_detection=True):
    class_names = [str(i) for i in range(10)]

    X = []
    y = []

    for class_idx, cls in enumerate(class_names):
        class_path = os.path.join(data_dir, cls)
        if not os.path.isdir(class_path):
            print(f"Warning: class folder not found, skipping: {class_path}")
            continue

        wav_files = glob(os.path.join(class_path, '*.wav'))
        print(f"Found {len(wav_files)} files for class {cls}")

        for wav_file in wav_files:
            try:
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

                feat_vec, feat_names = extract_features_from_frames(frames, method='statistical')
                X.append(feat_vec)
                y.append(class_idx)

            except Exception as e:
                print(f"  Error processing {wav_file}: {e}")
                continue

    if len(X) == 0:
        raise RuntimeError(f"No samples found in data_dir: {data_dir}")

    X = np.array(X)
    y = np.array(y)

    return X, y, feat_names, class_names


def main():
    parser = argparse.ArgumentParser(description='Train time-domain MLP for digit recognition')
    parser.add_argument('--data-dir', default=None, help='Dataset root (overrides env and config)')
    parser.add_argument('--results-dir', default='./results', help='Directory to save results/checkpoints')
    parser.add_argument('--window-type', default='hamming', choices=['rectangular', 'hamming', 'hanning'])
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--hidden-layers', nargs='*', type=int, default=None,
                        help='Hidden layer sizes, e.g. --hidden-layers 64 32')
    parser.add_argument('--save-name', default='mlp_checkpoint.pt', help='Checkpoint filename')

    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(2)

    results_dir = os.path.abspath(os.path.expanduser(args.results_dir))
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading dataset from: {data_dir}")
    X, y, feat_names, class_names = load_dataset(data_dir, window_type=args.window_type)

    print(f"Total samples: {len(X)}, feature dim: {X.shape[1]}")

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # normalize using training statistics
    X_train_norm, mean, std = normalize_features(X_train)
    X_test_norm, _, _ = normalize_features(X_test, mean, std)

    # prepare MLP trainer parameters
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

    # train (trainer.fit supports save_path or env SPEECH_REC_CPT)
    checkpoint_path = os.path.join(results_dir, args.save_name)

    print("Start training...")
    trainer.fit(X_train_norm, y_train, verbose=True, save_path=checkpoint_path, mean=mean, std=std, class_names=class_names)

    # evaluate
    results = trainer.evaluate(X_test_norm, y_test)

    print(f"Test accuracy: {results['accuracy']:.4f}")

    # save evaluation summary
    summary_path = os.path.join(results_dir, 'mlp_training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('MLP Training Summary\n')
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


if __name__ == '__main__':
    main()
