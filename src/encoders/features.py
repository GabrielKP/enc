import os
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from scipy.signal import hilbert, resample

from encoders.data import EMBEDDINGS_FILE, load_fmri, load_textgrid, load_wav
from encoders.utils import get_logger, load_config

log = get_logger(__name__)
cfg = load_config()

CACHE_DIR = Path(cfg["CACHE_DIR"])


def get_envelope(signal: np.ndarray) -> np.ndarray:
    """Compute the audio envelope"""
    log.info("Computing envelope.")
    return np.abs(hilbert(signal))  # type: ignore


def trim(signal: np.ndarray, sfreq: int, duration: float = 10.0) -> np.ndarray:
    log.info(f"Trimming beginning and end by {duration}s.")
    return signal[int(sfreq * duration) : int(-sfreq * duration)]


def downsample(
    signal: np.ndarray, sfreq: float, tr_len: float, n_trs: int
) -> np.ndarray:
    num_samples_uncorrected = signal.shape[0] / (sfreq * tr_len)
    # sometimes the signal samples do not match with the trs
    # either have to correct the number of resulting samples up or down
    # rounding won't work (`undertheinfluence`` vs `naked`)
    if num_samples_uncorrected > n_trs:
        num_samples = int(np.floor(num_samples_uncorrected))
    else:
        num_samples = int(np.ceil(num_samples_uncorrected))
    log.info(f"Downsampling to {num_samples} samples.")
    return resample(signal, num=num_samples)  # type: ignore


def load_envelope_data(
    story: str,
    tr_len: float,
    y_data: np.ndarray,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Load .wavfile, compute envelope, trim and downsample to match the
    number of samples in y_data.

    Parameters
    ----------
    story: str
        The story for which to load/compute the envelope.
    tr_len: float
        The time-to-repeat (TR)
    y_data: np.ndarray
        The data to match the sampling frequency of. Uses y_data.shape[0]
        to determine the number downsampling rate.
    use_cache: bool (default = True)
        Whether or not to save computed results to cache dir
        (will save to encoders.utils.load_config()['CACHE_DIR'])

    Returns
    -------
    np.ndarray
        The envelope data.

    """

    path_cache_x = Path(CACHE_DIR, "envelope_data", f"{story}_{tr_len}_X.npy")
    if Path.exists(path_cache_x) and use_cache:
        log.info(f"Loading from cache: {path_cache_x}")
        return np.load(path_cache_x)
    elif use_cache:
        log.info(f"No data found in cache: {path_cache_x}")

    n_trs = y_data.shape[0]

    sfreq, wav_data = load_wav(story)

    # if .wav array has two channel, take the mean
    if len(wav_data.shape) == 2:
        log.info("Wav has 2 channels, averaging across channel dimension.")
        wav_data = np.mean(wav_data, axis=1)

    X_envelope = get_envelope(wav_data)
    X_trimmed = trim(X_envelope, sfreq)
    X_data = downsample(X_trimmed, sfreq, tr_len, n_trs)
    X_data = X_data[:, np.newaxis]

    if use_cache:
        os.makedirs(Path(CACHE_DIR, "envelope_data"), exist_ok=True)
        np.save(path_cache_x, X_data)
        log.info("Cached results.")
    return X_data


# these tokens were defined in:
# https://github.com/HuthLab/deep-fMRI-dataset/blob/eaaa5cd186e0222c374f58adf29ed13ab66cc02a/encoding/ridge_utils/dsutils.py#L5C1-L5C96
SKIP_TOKENS = frozenset(
    ["sentence_start", "sentence_end", "{BR}", "{LG}", "{ls}", "{LS}", "{NS}", "sp"]
)


def load_embeddings() -> Tuple[np.ndarray, Dict]:
    """
    Load the embedding vectors and vocabulary from the EMBEDDINGS_FILE (h5py).
    """
    with h5py.File(EMBEDDINGS_FILE, "r") as f:
        # List all groups
        log.info(f"Loading: {EMBEDDINGS_FILE}")

        # Get the data
        data = np.array(f["data"])
        vocab = {e.decode("utf-8"): i for i, e in enumerate(np.array(f["vocab"]))}

        log.info(f"data shape: {data.shape}")
        log.info(f"vocab len: {len(vocab)}")

    return data, vocab


def get_embeddings(story: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings, vocabulary and word onset/offset times from the textgrid file.
    """
    vecs, vocab = load_embeddings()

    word_grid = load_textgrid(story)["word"]

    tokens = [
        row.text.lower()
        for _, row in word_grid.iterrows()
        if row.text not in SKIP_TOKENS
    ]
    starts = np.array(
        [row.start for _, row in word_grid.iterrows() if row.text not in SKIP_TOKENS]
    )
    stops = np.array(
        [row.stop for _, row in word_grid.iterrows() if row.text not in SKIP_TOKENS]
    )

    exist_tokens = [t for t in tokens if t in vocab]

    log.info(
        f"{len(exist_tokens)}/{len(tokens)}"
        + f" (missing {len(tokens) - len(exist_tokens)}) "
        + "story tokens found in vocab."
    )

    embs = np.array(
        [vecs[:, vocab[t]] if t in vocab else np.zeros(vecs.shape[0]) for t in tokens]
    )

    return embs, starts, stops


def load_sm1000_data(
    story: str, tr_len: float, y_data: np.ndarray, start_trim: float = 10.0
) -> np.ndarray:
    data, starts, stops = get_embeddings(story)

    # get 'mean word position'
    t_word = (starts + stops) / 2

    n_trs = y_data.shape[0]

    # get starting word index for each tr
    # first 10 seconds of the fMRI data is trimmed
    data_word_starts = [
        sum(t_word < idx_tr * tr_len + start_trim) for idx_tr in range(n_trs + 1)
    ]

    X_data = np.empty((n_trs, data.shape[1]))
    for idx_tr, (idx_start, idx_end) in enumerate(
        zip(data_word_starts[:-1], data_word_starts[1:])
    ):
        if idx_start == idx_end:
            # no word in tr
            X_data[idx_tr] = 0
        else:
            # average words occurring in tr
            X_data[idx_tr] = np.mean(data[idx_start:idx_end], axis=0)

    return X_data


def make_delayed(signal: np.ndarray, delays: np.ndarray, circpad=False) -> np.ndarray:
    """
    Create delayed versions of the 2-D signal.

    Parameters
    -----------
    signal : np.ndarray
        2-D array of shape (n_samples, n_features)
    delays : np.ndarray
        1-D array of delays to apply to the signal
        can be positive or negative; negative values advance the signal (shifting it
        backward)
    circpad : bool
        If True, use circular padding for delays
        If False, use zero padding for delays

    Returns
    --------
    np.ndarray
        2-D array of shape (n_samples, n_features * ndelays)
    """

    delayed_signals = []
    n_samples, n_features = signal.shape

    for delay in delays:
        delayed_signal = np.zeros_like(signal)
        if circpad:
            delayed_signal = np.roll(signal, delay, axis=0)
        else:
            if delay > 0:
                delayed_signal[delay:, :] = signal[:-delay, :]
            elif delay < 0:
                delayed_signal[:delay, :] = signal[-delay:, :]
            else:
                delayed_signal = signal.copy()
        delayed_signals.append(delayed_signal)

    return np.hstack(delayed_signals)


def sinc(f_c, t):
    """
    Sin function with cutoff frequency f_c.

    Parameters
    -----------
    f_c : float
        Cutoff frequency
    t : np.ndarray or float
        Time

    Returns
    --------
    np.ndarray or float
        Sin function with cutoff frequency f_c
    """
    return np.sin(np.pi * f_c * t) / (np.pi * f_c * t)


def lanczosfun(f_c, t, a=3):
    """
    Lanczos function with cutoff frequency f_c.

    Parameters
    -----------
    f_c : float
        Cutoff frequency
    t : np.ndarray or float
        Time
    a : int
        Number of lobes (window size), typically 2 or 3; only signals within the window
        will have non-zero weights.

    Returns
    --------
    np.ndarray or float
        Lanczos function with cutoff frequency f_c
    """
    val = sinc(f_c, t) * sinc(f_c, t / a)
    val[t == 0] = 1.0
    val[np.abs(t * f_c) > a] = 0.0

    return val


def lanczosinterp2D(signal, oldtime, newtime, window=3, cutoff_mult=1.0):
    """
    Lanczos interpolation for 2D signals; interpolates [signal] from [oldtime] to
    [newtime], assuming that the rows of [signal] correspond to [oldtime]. Returns a
    new signal with rows corresponding to [newtime] and the same number of columns as
    [signal].

    Parameters
    -----------
    signal : np.ndarray
        2-D array of shape (n_samples, n_features)
    oldtime : np.ndarray
        1-D array of old time points
    newtime : np.ndarray
        1-D array of new time points
    window : int
        Number of lobes (window size) for the Lanczos function
    cutoff_mult : float
        Multiplier for the cutoff frequency

    Returns
    --------
    np.ndarray
        2-D array of shape (len(newtime), n_features)
    """
    # Find the cutoff frequency
    f_c = 1 / (np.max(np.abs(np.diff(newtime)))) * cutoff_mult
    # Build the Lanczos interpolation matrix
    interp_matrix = np.zeros((len(newtime), len(oldtime)))
    for i, t in enumerate(newtime):
        interp_matrix[i, :] = lanczosfun(f_c, t - oldtime, a=window)
    # Interpolate the signal
    newsignal = np.dot(interp_matrix, signal)

    return newsignal


def downsample_embeddings_lanczos(
    embeddings: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    n_trs: int,
    tr_len: float,
    start_trim: float = 10.0,
) -> np.ndarray:
    word_times = (starts + stops) / 2
    tr_times = np.arange(n_trs) * tr_len + start_trim + tr_len / 2.0
    downsampled_embeddings = lanczosinterp2D(embeddings, word_times, tr_times, window=3)
    return downsampled_embeddings


def load_data_dict(
    stories: list[str],
    subject: str,
    feature: str,
    interpolation: str,
    shuffle: bool,
    tr_len: float,
    ndelays: int,
    use_cache: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    X_data_dict = dict()
    y_data_dict = dict()
    for story in stories:
        y_data = load_fmri(story, subject)
        if shuffle:
            y_data = np.random.permutation(y_data)

        if feature == "eng1000":
            if interpolation == "lanczos":
                data, starts, stops = get_embeddings(story)
                X_data = downsample_embeddings_lanczos(
                    data, starts, stops, y_data.shape[0], tr_len
                )
            elif interpolation == "average":
                X_data = load_sm1000_data(story, tr_len, y_data)
            X_data = make_delayed(X_data, np.arange(1, ndelays + 1), circpad=False)

        elif feature == "envelope":
            X_data = load_envelope_data(story, tr_len, y_data, use_cache)
            X_data = make_delayed(X_data, np.arange(1, ndelays + 1), circpad=False)

        assert X_data.shape[0] == y_data.shape[0], (
            f"X.shape={X_data.shape} and y.shape={y_data.shape} for {story} don't match"
        )

        X_data_dict[story] = X_data
        y_data_dict[story] = y_data

    return X_data_dict, y_data_dict
