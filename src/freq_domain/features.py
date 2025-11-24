"""
Frequency domain feature extraction: STFT, MFCC, GFCC, and dynamic features.
Core algorithms for Experiment 2.
"""
import torch
import numpy as np
import math
from scipy.signal import gammatone


def pre_emphasis(signal: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    """
    Pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]
    Boosts high frequency components.
    """
    return torch.cat([signal[0:1], signal[1:] - coeff * signal[:-1]])


def frame_signal(signal: torch.Tensor, frame_len: int, frame_shift: int) -> torch.Tensor:
    """
    Divide signal into overlapping frames.

    Args:
        signal: Input signal tensor
        frame_len: Frame length in samples
        frame_shift: Frame shift in samples

    Returns:
        Framed signal [num_frames, frame_len]
    """
    signal_len = len(signal)
    num_frames = int(np.ceil((signal_len - frame_len) / frame_shift)) + 1

    # Pad signal if needed
    pad_len = (num_frames - 1) * frame_shift + frame_len - signal_len
    if pad_len > 0:
        signal = torch.cat([signal, torch.zeros(pad_len, device=signal.device)])

    # Create frame indices
    indices = (torch.arange(frame_len, device=signal.device).unsqueeze(0) +
               torch.arange(0, num_frames * frame_shift, frame_shift, device=signal.device).unsqueeze(1))

    return signal[indices.long()]


def apply_window(frames: torch.Tensor, window_type: str = 'hamming') -> torch.Tensor:
    """
    Apply window function to frames.

    Args:
        frames: Input frames [num_frames, frame_len]
        window_type: 'hamming', 'hanning', or 'rectangular'

    Returns:
        Windowed frames
    """
    frame_len = frames.shape[1]
    n = torch.arange(frame_len, device=frames.device, dtype=frames.dtype)

    if window_type == 'hamming':
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (frame_len - 1))
    elif window_type == 'hanning':
        window = 0.5 * (1 - torch.cos(2 * math.pi * n / (frame_len - 1)))
    else:  # rectangular
        window = torch.ones(frame_len, device=frames.device)

    return frames * window


def compute_power_spectrum(frames: torch.Tensor, n_fft: int) -> torch.Tensor:
    """
    Compute power spectrum using FFT.

    Args:
        frames: Windowed frames [num_frames, frame_len]
        n_fft: FFT size

    Returns:
        Power spectrum [num_frames, n_fft//2 + 1]
    """
    # Zero-pad if needed
    if frames.shape[1] < n_fft:
        padding = torch.zeros(frames.shape[0], n_fft - frames.shape[1], device=frames.device)
        frames = torch.cat([frames, padding], dim=1)
    else:
        frames = frames[:, :n_fft]

    # FFT
    spectrum = torch.fft.rfft(frames, n=n_fft)
    power = torch.abs(spectrum) ** 2 / n_fft

    return power


def hz_to_mel(freq: float) -> float:
    """Convert Hz to Mel scale: 2595 * log10(1 + f/700)"""
    return 2595 * np.log10(1 + freq / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert Mel to Hz scale"""
    return 700 * (10 ** (mel / 2595.0) - 1)


def create_mel_filterbank(n_filters: int, n_fft: int, sample_rate: int,
                          device: str = 'cpu') -> torch.Tensor:
    """
    Create triangular Mel filterbank.

    Args:
        n_filters: Number of Mel filters
        n_fft: FFT size
        sample_rate: Audio sample rate

    Returns:
        Filterbank [n_filters, n_fft//2 + 1]
    """
    # Mel scale points
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert to FFT bins
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Create filterbank
    fbank = torch.zeros(n_filters, n_fft // 2 + 1, device=device)

    for m in range(1, n_filters + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        # Rising edge
        for k in range(left, center):
            fbank[m - 1, k] = (k - left) / (center - left)
        # Falling edge
        for k in range(center, right):
            fbank[m - 1, k] = (right - k) / (right - center)

    return fbank


def apply_dct(log_mel_energy: torch.Tensor, n_ceps: int) -> torch.Tensor:
    """
    Apply DCT to get MFCC coefficients.

    Args:
        log_mel_energy: Log Mel filterbank energy [num_frames, n_filters]
        n_ceps: Number of cepstral coefficients

    Returns:
        MFCC [num_frames, n_ceps]
    """
    num_frames, n_filters = log_mel_energy.shape
    device = log_mel_energy.device

    # Create DCT matrix
    n = torch.arange(n_ceps, device=device).float()
    m = torch.arange(n_filters, device=device).float()

    # DCT-II formula: cos(pi * n * (2m + 1) / (2M))
    dct_matrix = torch.cos(math.pi * n.unsqueeze(1) * (2 * m.unsqueeze(0) + 1) / (2 * n_filters))
    dct_matrix *= math.sqrt(2.0 / n_filters)

    return torch.matmul(log_mel_energy, dct_matrix.T)


def extract_mfcc(signal: np.ndarray, sample_rate: int,
                 n_mfcc: int = 13, n_fft: int = 512, n_filters: int = 26,
                 frame_len_ms: float = 25, frame_shift_ms: float = 10,
                 device: str = None) -> torch.Tensor:
    """
    Extract MFCC features from audio signal.

    Complete pipeline: Pre-emphasis -> Framing -> Windowing -> FFT ->
                       Mel Filterbank -> Log -> DCT -> MFCC

    Args:
        signal: Audio signal (numpy array)
        sample_rate: Sample rate in Hz
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT size
        n_filters: Number of Mel filters
        frame_len_ms: Frame length in milliseconds
        frame_shift_ms: Frame shift in milliseconds
        device: Computation device ('cpu', 'mps', 'cuda')

    Returns:
        MFCC features [num_frames, n_mfcc]
    """
    # Set device
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Convert to tensor
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal).float().to(device)
    else:
        signal = signal.float().to(device)

    # Calculate frame parameters
    frame_len = int(sample_rate * frame_len_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)

    # 1. Pre-emphasis
    signal = pre_emphasis(signal)

    # 2. Framing
    frames = frame_signal(signal, frame_len, frame_shift)

    # 3. Windowing
    frames = apply_window(frames, 'hamming')

    # 4. Power spectrum
    power_spec = compute_power_spectrum(frames, n_fft)

    # 5. Mel filterbank
    mel_fbank = create_mel_filterbank(n_filters, n_fft, sample_rate, device)
    mel_energy = torch.matmul(power_spec, mel_fbank.T)

    # 6. Log compression
    log_mel_energy = torch.log(mel_energy + 1e-10)

    # 7. DCT to get MFCC
    mfcc = apply_dct(log_mel_energy, n_mfcc)

    return mfcc


def compute_delta(features: torch.Tensor, window: int = 2) -> torch.Tensor:
    """
    Compute delta (first-order derivative) features.

    Uses regression formula:
    delta[t] = sum(n * (c[t+n] - c[t-n])) / (2 * sum(n^2))

    Args:
        features: Input features [num_frames, n_features]
        window: Window size for delta computation

    Returns:
        Delta features [num_frames, n_features]
    """
    num_frames, n_features = features.shape
    device = features.device

    # Pad features at boundaries
    padded = torch.cat([
        features[0:1].expand(window, -1),
        features,
        features[-1:].expand(window, -1)
    ], dim=0)

    # Compute denominator: 2 * sum(n^2)
    denominator = 2 * sum(n ** 2 for n in range(1, window + 1))

    # Compute delta
    delta = torch.zeros_like(features)
    for t in range(num_frames):
        for n in range(1, window + 1):
            delta[t] += n * (padded[t + window + n] - padded[t + window - n])
    delta /= denominator

    return delta


def compute_delta_delta(features: torch.Tensor, window: int = 2) -> torch.Tensor:
    """
    Compute delta-delta (second-order derivative / acceleration) features.

    Args:
        features: Input features [num_frames, n_features]
        window: Window size for delta computation

    Returns:
        Delta-delta features [num_frames, n_features]
    """
    delta = compute_delta(features, window)
    delta_delta = compute_delta(delta, window)
    return delta_delta


def extract_mfcc_with_delta(signal: np.ndarray, sample_rate: int,
                            n_mfcc: int = 13, n_fft: int = 512, n_filters: int = 26,
                            frame_len_ms: float = 25, frame_shift_ms: float = 10,
                            include_delta: bool = True, include_delta_delta: bool = True,
                            device: str = None) -> torch.Tensor:
    """
    Extract MFCC features with optional delta and delta-delta.

    Args:
        signal: Audio signal (numpy array)
        sample_rate: Sample rate in Hz
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT size
        n_filters: Number of Mel filters
        frame_len_ms: Frame length in milliseconds
        frame_shift_ms: Frame shift in milliseconds
        include_delta: Include first-order derivatives (13D -> 26D)
        include_delta_delta: Include second-order derivatives (26D -> 39D)
        device: Computation device

    Returns:
        MFCC features with delta [num_frames, n_mfcc * (1 + delta + delta_delta)]
    """
    # Extract base MFCC
    mfcc = extract_mfcc(
        signal, sample_rate,
        n_mfcc=n_mfcc, n_fft=n_fft, n_filters=n_filters,
        frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
        device=device
    )

    features = [mfcc]

    if include_delta:
        delta = compute_delta(mfcc)
        features.append(delta)

        if include_delta_delta:
            delta_delta = compute_delta_delta(mfcc)
            features.append(delta_delta)

    return torch.cat(features, dim=1)


def hz_to_erb(freq: float) -> float:
    """Convert Hz to ERB (Equivalent Rectangular Bandwidth) scale."""
    return 24.7 * (4.37 * freq / 1000 + 1)


def erb_to_hz(erb: float) -> float:
    """Convert ERB to Hz scale."""
    return (erb / 24.7 - 1) * 1000 / 4.37


def create_gammatone_filterbank(n_filters: int, n_fft: int, sample_rate: int,
                                 device: str = 'cpu') -> torch.Tensor:
    """
    Create Gammatone filterbank for GFCC extraction.

    Gammatone filters better model the human auditory system compared to Mel filters.

    Args:
        n_filters: Number of Gammatone filters
        n_fft: FFT size
        sample_rate: Audio sample rate

    Returns:
        Filterbank [n_filters, n_fft//2 + 1]
    """
    # Frequency range
    low_freq = 50  # Minimum frequency in Hz
    high_freq = sample_rate / 2

    # ERB scale center frequencies
    low_erb = 24.7 * np.log10(4.37 * low_freq / 1000 + 1)
    high_erb = 24.7 * np.log10(4.37 * high_freq / 1000 + 1)
    erb_points = np.linspace(low_erb, high_erb, n_filters)
    center_freqs = (10 ** (erb_points / 24.7) - 1) * 1000 / 4.37

    # FFT frequencies
    fft_freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)

    # Create filterbank
    fbank = np.zeros((n_filters, n_fft // 2 + 1))

    for i, cf in enumerate(center_freqs):
        # Equivalent Rectangular Bandwidth
        erb = 24.7 * (4.37 * cf / 1000 + 1)
        b = 1.019 * erb  # Bandwidth parameter

        # Gammatone magnitude response (simplified)
        # Using the magnitude response of a 4th order gammatone filter
        for j, f in enumerate(fft_freqs):
            if f > 0:
                # Simplified gammatone response
                t = (f - cf) / b
                fbank[i, j] = (1 + t ** 2) ** (-2)

        # Normalize each filter
        if np.sum(fbank[i]) > 0:
            fbank[i] /= np.sum(fbank[i])

    return torch.from_numpy(fbank).float().to(device)


def extract_gfcc(signal: np.ndarray, sample_rate: int,
                 n_gfcc: int = 13, n_fft: int = 512, n_filters: int = 26,
                 frame_len_ms: float = 25, frame_shift_ms: float = 10,
                 device: str = None) -> torch.Tensor:
    """
    Extract GFCC (Gammatone Frequency Cepstral Coefficients) features.

    Similar to MFCC but uses Gammatone filterbank instead of Mel filterbank.

    Args:
        signal: Audio signal (numpy array)
        sample_rate: Sample rate in Hz
        n_gfcc: Number of GFCC coefficients
        n_fft: FFT size
        n_filters: Number of Gammatone filters
        frame_len_ms: Frame length in milliseconds
        frame_shift_ms: Frame shift in milliseconds
        device: Computation device

    Returns:
        GFCC features [num_frames, n_gfcc]
    """
    # Set device
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Convert to tensor
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal).float().to(device)
    else:
        signal = signal.float().to(device)

    # Calculate frame parameters
    frame_len = int(sample_rate * frame_len_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)

    # 1. Pre-emphasis
    signal = pre_emphasis(signal)

    # 2. Framing
    frames = frame_signal(signal, frame_len, frame_shift)

    # 3. Windowing
    frames = apply_window(frames, 'hamming')

    # 4. Power spectrum
    power_spec = compute_power_spectrum(frames, n_fft)

    # 5. Gammatone filterbank (instead of Mel)
    gamma_fbank = create_gammatone_filterbank(n_filters, n_fft, sample_rate, device)
    gamma_energy = torch.matmul(power_spec, gamma_fbank.T)

    # 6. Log compression
    log_gamma_energy = torch.log(gamma_energy + 1e-10)

    # 7. DCT to get GFCC
    gfcc = apply_dct(log_gamma_energy, n_gfcc)

    return gfcc


def extract_gfcc_with_delta(signal: np.ndarray, sample_rate: int,
                            n_gfcc: int = 13, n_fft: int = 512, n_filters: int = 26,
                            frame_len_ms: float = 25, frame_shift_ms: float = 10,
                            include_delta: bool = True, include_delta_delta: bool = True,
                            device: str = None) -> torch.Tensor:
    """
    Extract GFCC features with optional delta and delta-delta.

    Args:
        signal: Audio signal (numpy array)
        sample_rate: Sample rate in Hz
        n_gfcc: Number of GFCC coefficients
        n_fft: FFT size
        n_filters: Number of Gammatone filters
        frame_len_ms: Frame length in milliseconds
        frame_shift_ms: Frame shift in milliseconds
        include_delta: Include first-order derivatives
        include_delta_delta: Include second-order derivatives
        device: Computation device

    Returns:
        GFCC features with delta [num_frames, n_gfcc * (1 + delta + delta_delta)]
    """
    # Extract base GFCC
    gfcc = extract_gfcc(
        signal, sample_rate,
        n_gfcc=n_gfcc, n_fft=n_fft, n_filters=n_filters,
        frame_len_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
        device=device
    )

    features = [gfcc]

    if include_delta:
        delta = compute_delta(gfcc)
        features.append(delta)

        if include_delta_delta:
            delta_delta = compute_delta_delta(gfcc)
            features.append(delta_delta)

    return torch.cat(features, dim=1)
