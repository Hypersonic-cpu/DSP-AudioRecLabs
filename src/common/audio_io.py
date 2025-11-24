"""
Audio I/O utilities for loading and saving audio files.
"""
import numpy as np
import scipy.io.wavfile as wav
import os


def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """
    Load audio file and return normalized signal.

    Args:
        filepath: Path to audio file (wav format)

    Returns:
        signal: Normalized audio signal in range [-1, 1]
        sample_rate: Sample rate in Hz
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    sample_rate, data = wav.read(filepath)

    # Convert to float and normalize
    if data.dtype == np.int16:
        signal = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        signal = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        signal = (data.astype(np.float32) - 128) / 128.0
    else:
        signal = data.astype(np.float32)

    # Convert stereo to mono
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    return signal, sample_rate


def save_audio(filepath: str, signal: np.ndarray, sample_rate: int):
    """
    Save audio signal to wav file.

    Args:
        filepath: Output file path
        signal: Audio signal (will be converted to int16)
        sample_rate: Sample rate in Hz
    """
    # Convert to int16
    if signal.max() <= 1.0 and signal.min() >= -1.0:
        signal_int = (signal * 32767).astype(np.int16)
    else:
        signal_int = signal.astype(np.int16)

    wav.write(filepath, sample_rate, signal_int)
