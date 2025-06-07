"""
module for loading fMRI data and features and the like
"""

import re
from pathlib import Path
from typing import Optional, Tuple, cast

import h5py
import numpy as np
import pandas as pd
from scipy.io import wavfile

from encoders.utils import get_logger, load_config

log = get_logger(__name__)
cfg = load_config()

DATADIR = Path(cfg["DATA_DIR"])
STORIES = cfg["STORIES"]
WAV_DIR = "stimuli"
EMBEDDINGS_FILE = Path("data", "lebel_data", "english1000sm.hf5")

TEXT_GRID_FORMATS = [
    'File type = "ooTextFile"',
    '"Praat chronological TextGrid text file"',
]


def load_wav(story: str) -> Tuple[int, np.ndarray]:
    """Load wav file. Return ndarray with shape [samples, channels]."""

    wav_path = Path(DATADIR, WAV_DIR, f"{story}.wav")
    sample_rate, wav = wavfile.read(wav_path)

    n_chans = wav.shape[1] if len(wav.shape) == 2 else 1

    log.info(
        f"{story}.wav | channels: {n_chans} | length {wav.shape[0] / sample_rate}s"
    )
    return sample_rate, wav


def parse_textgrid(text_grid_lines: list) -> dict:
    lines = [line.strip("\n").strip() for line in text_grid_lines]

    # find what textgrid format it is
    text_grid_ftype = lines[0]  # this is always the first line

    assert text_grid_ftype in TEXT_GRID_FORMATS, (
        f"Unexpected textgrid format: {text_grid_ftype[0]}"
    )

    if text_grid_ftype == TEXT_GRID_FORMATS[0]:
        """
        Example header:

        File type = "ooTextFile"
        Object class = "TextGrid"

        xmin = 0.0124716553288
        xmax = 729.993423837
        tiers? <exists>
        size = 2
        item []:
            item [1]:
                class = "IntervalTier"
                name = "phone"
                xmin = 0.0124716553288
                xmax = 729.993423837
                intervals: size = 6819
                intervals [1]:
                    xmin = 0.0124716553288
                    xmax = 2.82607699773
                    text = "sp"
                intervals [2]:
                    xmin = 2.82607709751
                    xmax = 2.9465576552097734
                    text = "S"
                intervals [3]:
                    xmin = 2.9465576552097734
                    xmax = 3.348726528529025
        """

        def find_tiers(tier_array: np.ndarray, is_long: bool):
            if is_long:
                starttimes = tier_array[1::4]
                stoptimes = tier_array[2::4]
                textstrings = tier_array[3::4]
            else:
                starttimes = tier_array[0::3]
                stoptimes = tier_array[1::3]
                textstrings = tier_array[2::3]

            starttimes = [float(e.strip("xmin = ").strip()) for e in starttimes]
            stoptimes = [float(e.strip("xmax = ").strip()) for e in stoptimes]
            textstrings = [e.strip("text = ").strip('"') for e in textstrings]

            assert len(starttimes) == len(stoptimes) == len(textstrings), (
                f"{len(starttimes)}, {len(stoptimes)}, {len(textstrings)}"
            )

            return starttimes, stoptimes, textstrings

        # check if it is a long or short TextGrid format
        is_long = "xmin = " in lines[3]  # xmin = 0.3 (long), just float is short

        # find mathc using re.match
        start = float(lines[3].strip("xmin = ").strip())
        stop = float(lines[4].strip("xmax = ").strip())

        assert stop > start

        # find information about the tiers by
        # matching the specific strings in file lines
        interval_tiers = []
        tier_names = {}
        tier_n = []
        tier_starts = []
        if is_long:
            for i, line in enumerate(lines):
                if re.match(r'class = "IntervalTier"', line):
                    interval_tiers.append(i)
                if re.match(r"name = ", line):
                    name = line.split('"')[1]
                    tier_names[name] = i
                if re.match(r"intervals: size = ", line):
                    tier_n.append(int(line.split(" ")[-1]))
                    tier_starts.append(i + 1)

                    # find which lines correspond to which tier
            phone_start, word_start = tier_starts
            phone_stop = phone_start + tier_n[0] * 4
            word_stop = word_start + tier_n[1] * 4
            phone_tier = np.array(lines[phone_start:phone_stop])
            word_tier = np.array(lines[word_start:word_stop])
        else:
            interval_tiers = cast(
                list[int], np.where(np.array(lines) == '"IntervalTier"')[0].tolist()
            )
            tier_names[lines[interval_tiers[0] + 1].replace('"', "")] = interval_tiers[
                0
            ]
            tier_names[lines[interval_tiers[1] + 1].replace('"', "")] = interval_tiers[
                1
            ]
            tier_starts = [i + 5 for i in interval_tiers]
            tier_n = []
            for i in interval_tiers:
                tier_n.append(int(lines[i + 4]))

                # find which lines correspond to which tier
            n_lines_per_entry = 3
            phone_start, word_start = tier_starts
            phone_stop = phone_start + (tier_n[0] * n_lines_per_entry)
            word_stop = word_start + (tier_n[1] * n_lines_per_entry)
            phone_tier = np.array(lines[phone_start:phone_stop])
            word_tier = np.array(lines[word_start:word_stop])

        phones_start, phones_stop, phones = find_tiers(phone_tier, is_long)
        phone_dict = {"start": phones_start, "stop": phones_stop, "text": phones}

        words_start, words_stop, words = find_tiers(word_tier, is_long)
        word_dict = {"start": words_start, "stop": words_stop, "text": words}

    elif text_grid_ftype == TEXT_GRID_FORMATS[1]:
        """Example header:
        "Praat chronological TextGrid text file"
        0.0124716553288 819.988889088   ! Time domain.
        2   ! Number of tiers.
        "IntervalTier" "phone" 0.0124716553288 819.988889088
        "IntervalTier" "word" 0.0124716553288 819.988889088
        1 0.0124716553288 1.26961451247
        "ns"
        2 0.0124716553288 1.26961451247
        "{NS}"
        1 1.26961451247 1.48948829731937
        "S"
        2 1.26961451247 2.23741496599
        "SO"
        """

        start = float(lines[1].split()[0])
        stop = float(lines[1].split()[1])

        n_tiers = int(lines[2].split()[0])
        tiername_lines = np.arange(3, 3 + n_tiers)
        tier_names = [lines[i].split()[1].strip('"') for i in tiername_lines]

        tier_indicators = np.array([int(line.split()[0]) for line in lines[5::2]])
        tier_indicators = np.repeat(tier_indicators, 2)

        times = lines[5::2]
        words = lines[6::2]
        assert len(times) == len(words), (
            f"Mismatch in number of elements ({len(times)}, {len(words)})"
        )

        phone_dict = {"start": [], "stop": [], "text": []}
        word_dict = {"start": [], "stop": [], "text": []}
        for t, w in zip(times, words):
            tier, start, stop = t.split()

            if tier == "1":
                phone_dict["start"].append(float(start))
                phone_dict["stop"].append(float(stop))
                phone_dict["text"].append(w.strip('"'))
            elif tier == "2":
                word_dict["start"].append(float(start))
                word_dict["stop"].append(float(stop))
                word_dict["text"].append(w.strip('"'))

    # put everything into a dataframe
    out: dict[str, Optional[pd.DataFrame]] = {n: None for n in ["phone", "word"]}

    out["phone"] = pd.DataFrame(phone_dict, columns=list(phone_dict.keys()))
    out["word"] = pd.DataFrame(word_dict, columns=list(word_dict.keys()))

    return out


def load_textgrid(story: str) -> dict[str, pd.DataFrame]:
    """
    Loads {story}.TextGrid from 'ds003020/derivative/TextGrids' folder.

    Parameters
    ----------
    story: str
        Story to load

    Returns
    -------
    dict
        Dictionary with keys 'phone' and 'word', each containing a dataframe
        with phone and word onset times, respectfully.

    """
    textgrid_dir = DATADIR / "derivative" / "TextGrids"
    fn = textgrid_dir / f"{story}.TextGrid"

    with open(fn, "r") as f:
        lines = f.readlines()

    word_phone_dict = parse_textgrid(lines)

    return word_phone_dict  # type: ignore


def load_fmri(story: str, subject: str) -> np.ndarray:
    """Load fMRI data. Return ndarray with shape [time, voxels]."""

    subject_dir = Path(DATADIR, f"derivative/preprocessed_data/{subject}")
    resp_path = Path(subject_dir, f"{story}.hf5")
    hf = h5py.File(resp_path, "r")
    log.info(
        f"{story}.hf5"
        f" | {subject}"
        f" | time: {hf['data'].shape[0]}"  # type: ignore
        f" | voxels: {hf['data'].shape[1]}"  # type: ignore
    )
    return np.array(hf["data"][:])  # type: ignore
