# https://github.com/openvpi/audio-slicer/blob/main/slicer2.py


import librosa
import numpy as np
from typing import Callable, Union
from fade import apply_hann_edge


def _signal_to_frame_nonsilent(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
    top_db: float = 60,
    ref: Union[Callable, float] = np.max,
    aggregate: Callable = np.max,
) -> np.ndarray:
    """Frame-wise non-silent indicator for audio input.

    This is a helper function for `trim` and `split`.

    Parameters
    ----------
    y : np.ndarray
        Audio signal, mono or stereo

    frame_length : int > 0
        The number of samples per frame

    hop_length : int > 0
        The number of samples between frames

    top_db : number
        The threshold (in decibels) below reference to consider as
        silence.
        You can also use a negative value for `top_db` to treat any value
        below `ref + |top_db|` as silent.  This will only make sense if
        `ref` is not `np.max`.

    ref : callable or float
        The reference amplitude

    aggregate : callable [default: np.max]
        Function to aggregate dB measurements across channels (if y.ndim > 1)

        Note: for multiple leading axes, this is performed using ``np.apply_over_axes``.

    Returns
    -------
    non_silent : np.ndarray, shape=(m,), dtype=bool
        Indicator of non-silent frames
    """
    # Compute the MSE for the signal
    mse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

    # Convert to decibels and slice out the mse channel
    db: np.ndarray = librosa.core.amplitude_to_db(mse[..., 0, :], ref=ref, top_db=None)

    # Aggregate everything but the time dimension
    if db.ndim > 1:
        db = np.apply_over_axes(aggregate, db, range(db.ndim - 1))
        # Squeeze out leading singleton dimensions here
        # We always want to keep the trailing dimension though
        db = np.squeeze(db, axis=tuple(range(db.ndim - 1)))

    return db > -top_db


def split_silent_librosa(
    y: np.ndarray,
    sr: int,
    min_silence_len: int = 10,
    seek_step: int = 1,
    top_db: int = 60,
) -> np.ndarray:
    """Split an audio signal into silent intervals using Librosa.

    Parameters
    ----------
    y : np.ndarray, shape=(..., n)
        An audio signal. Multi-channel is supported.
    min_silence_len : number > 0
        The minimum length for any silent section in ms.
    seek_step : number > 0
        The step size for interating over the segment in ms.
    top_db : number > 0
        The threshold (in decibels) below reference to consider as silence.

    Returns
    -------
    intervals : np.ndarray, shape=(m, 2)
        ``intervals[i] == (start_i, end_i)`` are the start and end time
        (in samples) of silent interval ``i``.
    """

    # pydub works in ms; convert to samples
    frame_length = max(1, int(round(min_silence_len * sr / 1000)))
    hop_length = max(1, int(round(seek_step * sr / 1000)))

    non_silent = _signal_to_frame_nonsilent(
        y,
        frame_length=frame_length,
        hop_length=hop_length,
        top_db=top_db,
    )
    # Flip to silent
    silent = ~non_silent

    # Interval slicing, adapted from
    # https://stackoverflow.com/questions/2619413/efficiently-finding-the-interval-with-non-zeros-in-scipy-numpy-in-python
    # Find points where the sign flips
    edges = np.flatnonzero(np.diff(silent.astype(int)))

    # Pad back the sample lost in the diff
    edges = [edges + 1]

    # If the first frame had high energy, count it
    if silent[0]:
        edges.insert(0, np.array([0]))

    # Likewise for the last frame
    if silent[-1]:
        edges.append(np.array([len(silent)]))

    # Convert from frames to samples
    edges = librosa.core.frames_to_samples(np.concatenate(edges), hop_length=hop_length)

    # Clip to the signal duration
    edges = np.minimum(edges, y.shape[-1])

    # Stack the results back as an ndarray
    edges = edges.reshape((-1, 2))  # type: np.ndarray
    return edges


def split_silent_pydub(
    y: np.ndarray,
    sr: int,
    min_silence_len: int = 10,
    seek_step: int = 1,
    top_db: int = 60,
):
    """Split an audio signal into silent intervals using PyDub.

    Parameters
    ----------
    y : np.ndarray, shape=(..., n)
        An audio signal. Multi-channel is supported.
    min_silence_len : number > 0
        The minimum length for any silent section in ms.
    seek_step : number > 0
        The step size for interating over the segment in ms.
    top_db : number > 0
        The threshold (in decibels) below reference to consider as silence.

    Returns
    -------
    intervals : np.ndarray, shape=(m, 2)
        ``intervals[i] == (start_i, end_i)`` are the start and end time
        (in samples) of silent interval ``i``.
    """

    from pydub import AudioSegment
    from pydub.silence import detect_silence
    
    # _to_sr = lambda ms, sr: int(ms * sr / 1000)
    _to_sr = lambda ms, sr: int(round(ms * sr / 1000))

    wav_int = np.clip(np.round(y * 32768), -32768, 32767).astype(np.int16)
    audio_segment = AudioSegment(
        data=wav_int.tobytes(),
        frame_rate=int(sr),
        sample_width=wav_int.dtype.itemsize,
        channels=1,
    )

    silence_thresh = audio_segment.max_dBFS - top_db
    silent_ranges = detect_silence(
        audio_segment,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=seek_step,
    )

    return np.array([[_to_sr(s, sr), _to_sr(e, sr)] for s, e in silent_ranges])


class EvaluateSilence:
    def __init__(
        self,
        y: np.ndarray,
        sr: int,
        pydub: bool = False,
        min_silence_len: int = 10,
        seek_step: int = 1,
        top_db: int = 60,
    ) -> None:
        
        common_kwargs = dict(
            y = y,
            sr = int(sr),
            min_silence_len = int(min_silence_len),
            seek_step = int(seek_step),
            top_db = int(abs(top_db)),
        )
        if pydub:
            print("Using PyDub...")
            self.intervals = split_silent_pydub(**common_kwargs)
        else:
            print("Using Librosa...")
            self.intervals = split_silent_librosa(**common_kwargs)

    def normalize_boundary(
        self,
        end: int,
        start: int,
        neg_index: int = 0,
    ) -> tuple[int | None, int | None]:
        
        if start <= end:
            return end - neg_index, start - neg_index
        
        candidates = self.intervals[
            (self.intervals[:, 1] > end) & (self.intervals[:, 0] < start)
        ]

        if len(candidates):
            candidates = np.column_stack([
                np.maximum(candidates[:, 0], end),
                np.minimum(candidates[:, 1], start),
            ])
            best = np.argmax(candidates[:, 1] - candidates[:, 0])
            true_end, true_start = candidates[best]
        
            return true_end - neg_index, true_start - neg_index
        
        else:
            return None, None


def samples_to_db(y: np.ndarray, db_floor: int = -145):
    """Converts waveform to dB"""
    eps = 10 ** (db_floor / 20)
    return 20 * np.log10(np.maximum(np.abs(y), eps))


# def samples_to_rms_db(y: np.ndarray, sr: int, window_ms: int = 10, db_floor: int = -145):
#     """Converts waveform to smoothed RMS dB"""
#     window_size = max(1, int(sr * window_ms / 1000))
#     window = np.ones(window_size) / window_size
#     power_env = np.convolve(y ** 2, window, mode = "same")
#     rms_env = np.sqrt(np.maximum(power_env, 0))
#     return samples_to_db(rms_env, db_floor)
# def find_true_boundary(y, end, start, sr = None, threshold_db = None):
#     if start <= end:
#         return end, start
#     db_vals = samples_to_rms_db(y[end:start], sr)
#     # db_vals = samples_to_db(y[end:start])
#     # if sr is not None:
#     #     db_rms = samples_to_rms_db(y[end:start], sr)
#     #     db_vals = np.maximum(db_vals, db_rms)
#     if threshold_db is None:
#         below_thresh = np.array([], dtype = np.int64)
#     else:
#         below_thresh = np.where(db_vals <= threshold_db)[0]
#     if len(below_thresh) == 0:
#         threshold_db = np.percentile(db_vals, 50)
#         threshold_db = np.clip(threshold_db, -90, -60)
#         below_thresh = np.where(db_vals <= threshold_db)[0]
#         if len(below_thresh) == 0:
#             threshold_db = db_vals.min()
#             below_thresh = np.where(db_vals <= threshold_db)[0]
#         print(threshold_db)
#     below_thresh = end + below_thresh
#     breaks = np.where(np.diff(below_thresh) > 1)[0]
#     run_starts = np.r_[0, breaks + 1]
#     run_ends = np.r_[breaks, len(below_thresh) - 1]
#     lengths = below_thresh[run_ends] - below_thresh[run_starts] + 1
#     best = np.argmax(lengths)
#     true_end = below_thresh[run_starts[best]]
#     true_start = below_thresh[run_ends[best]] + 1
#     return true_end, true_start


def reduce_silence(
    wav: np.ndarray,
    sr: int,
    alignment: list[dict],
    pydub: bool = False,
    max_pause_secs: float = 0.2,
    max_clause_secs: float = 0.4,
    max_sent_secs: float = 0.6,
):
    
    max_pause = int(max_pause_secs * sr)
    max_clause = int(max_clause_secs * sr)
    max_sent = int(max_sent_secs * sr)

    max_pause_half = int(max_pause // 2)
    max_clause_half = int(max_clause // 2)
    max_sent_half = int(max_sent // 2)

    wav_helper = EvaluateSilence(wav, sr, pydub = pydub)

    neg_index = 0
    for i in range(len(alignment)-1):
        part = alignment[i]
        part_n = alignment[i+1]
        end = int(part["end"] * sr)
        start = int(part_n["start"] * sr)
        end, start = wav_helper.normalize_boundary(end, start, neg_index)
        if end is None or (diff := start - end) <= 0:
            continue

        text = part["word"]
        is_pause, is_clause, is_sent = classify_text_end(text)

        end_idx = start_idx = None
        if is_pause and diff > max_pause:
            end_idx = end + max_pause_half
            start_idx = start - max_pause_half

        elif is_clause and diff > max_clause:
            end_idx = end + max_clause_half
            start_idx = start - max_clause_half

        elif is_sent and diff > max_sent:
            end_idx = end + max_sent_half
            start_idx = start - max_sent_half

        if end_idx is not None and start_idx is not None:
            wav_a = wav[:end_idx]
            wav_b = wav[start_idx:]
            wav_a = apply_hann_edge(y = wav_a, sr = sr, fade_ms = 2)
            wav_b = apply_hann_edge(y = wav_b, sr = sr, fade_ms = 2)
            neg_length = len(wav) - (len(wav_a) + len(wav_b))
            neg_index += neg_length
            wav = np.concatenate([wav_a, wav_b])
            print("Difference:", diff / sr, "|", "Removed:", neg_length / sr)
            print(part)
            print(part_n)
            print("---")

    return wav
