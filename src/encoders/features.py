from typing import Dict, Tuple

import h5py
import numpy as np
from scipy.signal import hilbert, resample

from encoders.data import EMBEDDINGS_FILE, load_textgrid
from encoders.utils import get_logger

log = get_logger(__name__)


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
        f"{len(exist_tokens)}/{len(tokens)} (missing {len(tokens) - len(exist_tokens)}) story tokens found in vocab."
    )

    embs = np.array(
        [vecs[:, vocab[t]] if t in vocab else np.zeros(vecs.shape[0]) for t in tokens]
    )

    return embs, starts, stops
