#!/usr/bin/env python3
"""
Real-time single-file inference script.

Usage example:
  python run_time_single.py --file ./data/0/sample.wav --params ./results/mlp_checkpoint.pt

The checkpoint expected format (torch.save) is a dict containing at least:
  - 'model_state_dict': state dict of the MLP model
  - 'input_size': int
  - 'hidden_layers': list of ints
  - 'num_classes': int
  - 'mean': array-like (feature mean used during training)
  - 'std': array-like (feature std used during training)
  - optionally 'class_names': list of class labels (folder names)

If you train the model yourself, save the checkpoint like:
  torch.save({
      'model_state_dict': trainer.model.state_dict(),
      'input_size': input_size,
      'hidden_layers': hidden_layers,
      'num_classes': num_classes,
      'mean': mean, 'std': std,
      'class_names': class_names
  }, 'path/to/mlp_checkpoint.pt')
"""
import os
import sys
import argparse
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Run single-file digit recognition (time-domain MLP)')

    parser.add_argument('--file', '-f', required=True, help='Path to WAV file to recognize')
    parser.add_argument('--params', '-p', required=True, help='Path to saved MLP checkpoint (.pt)')
    parser.add_argument('--window-type', default='hamming', choices=['rectangular', 'hamming', 'hanning'], help='Window type for framing')

    args = parser.parse_args()

    # Ensure project root on path so imports work like other scripts
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    import config
    # audio processing and feature extraction
    from src.time_domain.audio_processing import process_audio_file
    from src.time_domain.feature_extraction import extract_features_from_frames
    from src.time_domain.models import MLPClassifier

    wav_path = os.path.abspath(os.path.expanduser(args.file))
    params_path = os.path.abspath(os.path.expanduser(args.params))

    if not os.path.exists(wav_path):
        print(f"Error: WAV file not found: {wav_path}")
        sys.exit(2)

    if not os.path.exists(params_path):
        print(f"Error: params file not found: {params_path}")
        sys.exit(2)

    # 1) Process audio (endpoint detection, framing)
    frames, sample_rate, metadata = process_audio_file(
        wav_path,
        frame_length=config.FRAME_LENGTH,
        frame_shift=config.FRAME_SHIFT,
        window_type=args.window_type,
        do_endpoint_detection=True,
        energy_high_ratio=config.ENERGY_HIGH_RATIO,
        energy_low_ratio=config.ENERGY_LOW_RATIO,
        zcr_threshold_ratio=config.ZCR_THRESHOLD_RATIO
    )

    # 2) Extract statistical features (same as training)
    feature_vector, feature_names = extract_features_from_frames(frames, method='statistical')

    # 3) Load checkpoint
    checkpoint = torch.load(params_path, map_location='cpu')

    # Expect checkpoint to contain model architecture info and normalization stats
    required_keys = ['model_state_dict', 'input_size', 'hidden_layers', 'num_classes']
    for k in required_keys:
        if k not in checkpoint:
            print(f"Error: checkpoint missing required key: {k}")
            sys.exit(3)

    mean = checkpoint.get('mean', None)
    std = checkpoint.get('std', None)

    if mean is None or std is None:
        print("Warning: checkpoint does not include 'mean'/'std'. Using sample statistics (may mismatch training).")
        feat = feature_vector.astype(np.float32)
        mean = np.mean(feat, axis=0)
        std = np.std(feat, axis=0)
        std = np.where(std == 0, 1.0, std)

    # Normalize
    feature_norm = (feature_vector - mean) / std

    # 4) Recreate model and load weights
    model = MLPClassifier(checkpoint['input_size'], checkpoint['hidden_layers'], checkpoint['num_classes'])
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        # If the checkpoint contains a full model (saved module), try loading directly
        print(f"Loading state_dict failed: {e}")
        # attempt to load whole-object
        try:
            obj = checkpoint.get('model', None)
            if obj is not None and hasattr(obj, 'state_dict'):
                model = obj
            else:
                print("Failed to recover model from checkpoint.")
                raise
        except Exception:
            raise

    model.eval()

    # 5) Run inference
    with torch.no_grad():
        x = torch.FloatTensor(feature_norm).unsqueeze(0)  # shape (1, num_features)
        outputs = model(x)
        _, pred = torch.max(outputs, dim=1)
        pred = int(pred.item())

    # Optional class names
    class_names = checkpoint.get('class_names', None)

    if class_names and isinstance(class_names, (list, tuple)) and len(class_names) > pred:
        label = class_names[pred]
    else:
        label = str(pred)

    print(f"Prediction: {label} (class index: {pred})")


if __name__ == '__main__':
    main()
