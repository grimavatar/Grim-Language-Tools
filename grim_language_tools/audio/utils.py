import numpy as np
import soundfile as sf
from pathlib import Path
from resample import resample


def load_audio(audio: str | Path | tuple[np.ndarray, int], target_sr: int = None) -> np.ndarray:
    if isinstance(audio, (str, Path)):
        wav, sr = sf.read(audio, dtype = "float64", always_2d = False)
    else:
        wav, sr = audio
        if not (isinstance(wav, np.ndarray) and isinstance(sr, int)):
            raise ValueError(f"'audio' must be a str, Path, or tuple of (np.ndarray, int), but got ({type(wav).__name__}, {type(sr).__name__})")
    if target_sr is not None and sr != target_sr:
        wav = resample(y = wav, orig_sr = int(sr), target_sr = target_sr)
        sr = int(target_sr)
    return wav, sr


def get_duration(audio: str | Path | tuple[np.ndarray, int]) -> float:
    """Get duration in secs"""
    wav, sr = load_audio(audio)
    return wav.shape[0] / sr

def get_max_duration(audio: list[str | Path | tuple[np.ndarray, int]]) -> float:
    if not isinstance(audio, list):
        audio = [audio]
    return max(get_duration(e) for e in audio)
