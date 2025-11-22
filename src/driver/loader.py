import numpy as np
import wave

def load_wav(filepath: str) -> tuple[np.ndarray, int] :
    """
    Returns:
        audio_data: Audio data, normalized to [-1, +1]
        sample_rate: Sampling rate
    """
    with wave.open(filepath, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        audio_bytes = wav_file.readframes(n_frames)

        # 转换为numpy数组
        if sample_width == 1:
            dtype = np.uint8
            audio_data = np.frombuffer(audio_bytes, dtype=dtype)
            audio_data = (audio_data - 128) / 128.0
        elif sample_width == 2:
            dtype = np.int16
            audio_data = np.frombuffer(audio_bytes, dtype=dtype)
            audio_data = audio_data / 32768.0
        else:
            raise ValueError(f"不支持的采样位数: {sample_width}")

        # 如果是立体声，转换为单声道
        if n_channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)

    return audio_data, sample_rate
