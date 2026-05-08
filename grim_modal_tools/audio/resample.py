import sys
import soxr
import numpy as np
import soundfile as sf


def resample(y: np.ndarray, orig_sr: float, target_sr: float, res_type: str = "soxr_hq", fix: bool = True, axis: int = -1):
    def _fix_length(data: np.ndarray, *, size: int, axis: int = -1) -> np.ndarray:
        n = data.shape[axis]
        if n > size:
            slices = [slice(None)] * data.ndim
            slices[axis] = slice(0, size)
            return data[tuple(slices)]
        elif n < size:
            lengths = [(0, 0)] * data.ndim
            lengths[axis] = (0, size - n)
            return np.pad(data, lengths, mode = "constant")

        return data

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr
    n_samples = int(np.ceil(y.shape[axis] * ratio))

    y_hat = np.apply_along_axis(
        soxr.resample, axis = axis, arr = y, in_rate = orig_sr, out_rate = target_sr, quality = res_type
    )

    if fix:
        y_hat = _fix_length(y_hat, size = n_samples, axis = axis)
    
    return np.asarray(y_hat, dtype = y.dtype)  # Match dtypes


def cli_resample(input_path: str, target_sr: int, replace: bool = False) -> str:
    if replace:
        output_path = input_path
    else:
        output_path = input_path.rsplit(".", 1)[0] + f".{str(target_sr)[:2]}k.wav"
    
    wav, sr = sf.read(input_path, dtype = "float64", always_2d = False)
    # wav, sr = librosa.load(input_path, sr = None, mono = True)

    wav_resample = resample(y = wav, orig_sr = int(sr), target_sr = target_sr)

    sf.write(output_path, wav_resample, target_sr)

    return output_path


if __name__ == "__main__":

    input_path = "audio"

    if len(sys.argv) > 1 and sys.argv[1].strip():
        input_path = sys.argv[1].strip()

    target_sr = 24000

    if len(sys.argv) == 3 and sys.argv[2].strip():
        target_sr = int(round(float(sys.argv[2].strip()) * 1000))

    cli_resample(input_path, target_sr)
