import sys
import numpy as np
import soundfile as sf


def apply_hann_edge(
    y: np.ndarray,
    sr: float | int,
    fade_ms: int = 2,
    start: bool | None = None,
    end: bool | None = None,
) -> np.ndarray:
    fade_len = int(sr * fade_ms / 1000)
    fade_len = min(fade_len, len(y) // 2)
    hann = np.hanning(2 * fade_len)
    
    if start or y[0] != 0:
        y[:fade_len] *= hann[:fade_len, None] if y.ndim > 1 else hann[:fade_len]
    if end or y[-1] != 0:
        y[-fade_len:] *= hann[fade_len:, None] if y.ndim > 1 else hann[fade_len:]
    
    return y


if __name__ == "__main__":

    input_path = "audio"

    if len(sys.argv) > 1 and sys.argv[1].strip():
        input_path = sys.argv[1].strip()

    output_path = input_path.rsplit(".", 1)[0] + ".hann.wav"

    wav, sr = sf.read(input_path, dtype = "float64", always_2d = False)
    # wav, sr = librosa.load(input_path, sr = None, mono = True)

    output_wav = apply_hann_edge(y = wav, sr = sr, fade_ms = 2)

    sf.write(output_path, output_wav, sr)
