import argparse
import os
from pathlib import Path
from typing import Optional, Union

import cortex
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from encoders.data import load_fmri, load_wav
from encoders.features import downsample, get_embeddings, get_envelope, trim
from encoders.regression import pearsonr, ridge_regression, ridge_regression_huth
from encoders.utils import get_logger, lanczosinterp2D, load_config, make_delayed

log = get_logger(__name__)
cfg = load_config()

DATADIR = Path(cfg["DATA_DIR"])
STORIES = cfg["STORIES"]
WAV_DIR = "stimuli"
CACHE_DIR = Path(cfg["CACHE_DIR"])


def load_envelope_data(
    story: str,
    tr_len: float,
    y_data: np.ndarray,
    use_cache: bool = True,
) -> np.ndarray:
    path_cache_x = Path(CACHE_DIR, "envelope_data", f"{story}_{tr_len}_X.npy")
    if Path.exists(path_cache_x) and use_cache:
        log.info(f"Loading from cache: {path_cache_x}")
        return np.load(path_cache_x)
    elif use_cache:
        log.info(f"No data found in cache: {path_cache_x}")

    n_trs = y_data.shape[0]

    sfreq, wav_data = load_wav(story)

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
            # average words occuring in tr
            X_data[idx_tr] = np.mean(data[idx_start:idx_end], axis=0)

    return X_data


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
    predictor: str,
    interpolation: str,
    shuffle: bool,
    tr_len: float,
    n_delays: int,
    use_cache: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    X_data_dict = dict()
    y_data_dict = dict()
    for story in stories:
        y_data = load_fmri(story, subject)
        if shuffle:
            y_data = np.random.permutation(y_data)

        if predictor == "embeddings":
            if interpolation == "lanczos":
                data, starts, stops = get_embeddings(story)
                X_data = downsample_embeddings_lanczos(
                    data, starts, stops, y_data.shape[0], tr_len
                )
            elif interpolation == "average":
                X_data = load_sm1000_data(story, tr_len, y_data)
            X_data = make_delayed(X_data, np.arange(1, n_delays + 1), circpad=False)

        elif predictor == "envelope":
            X_data = load_envelope_data(story, tr_len, y_data, use_cache)
            X_data = make_delayed(X_data, np.arange(1, n_delays + 1), circpad=False)

        elif predictor == "embeddings_huth":
            X_data = np.load(f"/Volumes/opt/enc/data/embeddings_huth/{story}.npy")
            X_data = make_delayed(X_data, np.arange(1, n_delays + 1), circpad=False)

        assert X_data.shape[0] == y_data.shape[0], (
            f"X.shape={X_data.shape} and y.shape={y_data.shape} for {story} don't match"
        )

        X_data_dict[story] = X_data
        y_data_dict[story] = y_data

    return X_data_dict, y_data_dict


def do_loocv_regression(
    predictor: str = "embeddings",
    stories: list[str] = STORIES,
    n_train_stories: Optional[int] = None,
    subject: str = "UTS02",
    tr_len: float = 2.0,
    n_delays: int = 4,
    interpolation: str = "lanczos",
    ridge_implementation: str = "ridge_huth",
    alphas: Optional[np.ndarray] = None,
    use_cache: bool = True,
    shuffle: bool = False,
) -> tuple[
    np.ndarray, list[np.ndarray], list[np.ndarray], list[Union[float, np.ndarray]]
]:
    # 1. choose stories
    # the order of stories is determined in config.yaml or via the argument.
    if n_train_stories is None:
        n_train_stories = len(stories) - 1
    stories = stories[: (n_train_stories + 1)]

    # 2. load all data
    X_data_dict, y_data_dict = load_data_dict(
        stories,
        subject,
        predictor,
        interpolation,
        shuffle,
        tr_len,
        n_delays,
        use_cache,
    )

    kf = KFold(n_splits=len(stories))

    # result arrays
    scores_list = list()
    weights_list = list()
    best_alphas_list = list()
    for fold, (train_indices, test_indices) in enumerate(kf.split(stories)):
        log.info(f"Fold {fold}")

        curr_train_stories = [stories[idx] for idx in train_indices]
        curr_test_stories = [stories[idx] for idx in test_indices]

        log.info(f"{fold} | Running Regression")
        if ridge_implementation == "ridge_huth":
            scores, weights, best_alphas = ridge_regression_huth(
                train_stories=curr_train_stories,
                test_stories=curr_test_stories,
                X_data_dict=X_data_dict,
                y_data_dict=y_data_dict,
                score_fct=pearsonr,  # type: ignore
                alphas=alphas,
            )
        else:
            scores, weights, best_alphas = ridge_regression(
                train_stories=curr_train_stories,
                test_stories=curr_test_stories,
                X_data_dict=X_data_dict,
                y_data_dict=y_data_dict,
                score_fct=pearsonr,  # type: ignore
                alphas=alphas,
            )
        log.info(f"{fold} | Mean corr: {scores.mean()}")
        log.info(f"{fold} | Max corr : {scores.max()}")

        scores_list.append(scores)
        weights_list.append(weights)
        best_alphas_list.append(best_alphas)

    # aggregate scores
    mean_scores = np.mean(scores_list, axis=0)

    return mean_scores, scores_list, weights_list, best_alphas_list


def do_simple_regression(
    predictor: str = "embeddings",
    stories: list[str] = STORIES,
    n_train_stories: Optional[int] = None,
    test_story: Optional[str] = None,
    n_repeats: int = 5,
    subject: str = "UTS02",
    tr_len: float = 2.0,
    n_delays: int = 4,
    interpolation: str = "lanczos",
    ridge_implementation: str = "ridge_huth",
    use_cache: bool = True,
    shuffle: bool = False,
    seed: Optional[int] = 123,
    keep_train_stories_in_mem: bool = True,
) -> tuple[
    np.ndarray, list[np.ndarray], list[np.ndarray], list[Union[float, np.ndarray]]
]:
    """Run regression for n_repeats and return results.

    Parameters
    ----------
    predictor : {"envelope", "embeddings", "embeddings_huth"}
        Which predictor to use for the regression. "envelope": audio envelope that
        participants heard. "embeddings": embeddings computed from the words that
        participants heard. "embeddings_huth": embeddings which were used in the
        huth paper (https://www.nature.com/articles/s41597-023-02437-z)
    stories: list[str], default=STORIES
        Pool of stories, the default is determined by the config.yaml
    n_train_stories: int, optional
        The amount of training stories sampled from `stories`, if this is `None` will
        use all except one story.
    test_story : str or `None`, default=`None`
        The story on which the regression models will be tested.
        If `None`, the test story will be randomly selected from the pool of stories.
    n_repeats : int, default=5
        If `strategy="simple"`, determines how often regression is repeated on a
        different train/test set.
    subject : {"UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06", "UTS07", "UTS08"},
        default="UTS02"
        Subject identifier
    tr_len: float, default=2.0
        Length of tr-windows used to sample fMRI data.
    n_delays: int, default=4
        How many delays are used to model the HRF, which is modeled by adding
        a shifted set of duplicated features for each delay. `n_delays=5` implies
        that the the features of the stimulus are shifted concatinated 5 times
        to training/testing data.
    interpolation : {"lanczos", "average"}, default="lanczos"
        Whether to use lanczos interpolation or just average the words within a TR.
        Only applies to the 'embeddings' predictor.
    ridge_implementation : {"ridgeCV", "ridge_huth"}, default="ridge_huth"
        Which implementation of ridge regression to use. "ridgeCV" uses the RidgeCV
        function from sklearn.
        "ridge_huth" uses the implementation from the Huth lab codebase which applies
        SVD to the data matrix and computes correlation scores with bootstrapping.
    alphas : np.ndarray or `None`, default = `None`
        Array of alpha values to optimize over. If `None`, will choose
        default value of the regression function.
    use_cache: bool, default=True
        Whether the cache is used for `envelope` features.
    shuffle: bool, default=False
        Whether to shuffle the predictors (features).
    seed: int | None, default=123
        Seed determining sampling of stories.
    keep_train_stories_in_mem: bool, default=True
        Whether stories are kept in memory after first loading. Unless when using all
        stories turning this off will reduce the memory footprint, but increase the
        time is spent loading data.

    Returns
    -------
    mean_scores : np.ndarray
        The mean prediction scores for each repeat.
    scores_list : list of np.ndarray
        The scores for each repeat.
    weights : list of np.ndarray
        The regression weights for each repeat.
    alphas : list of float or list of np.ndarray
        The best alphas for each voxel, for each repeat.
    """

    if n_train_stories is None:
        n_train_stories = len(stories) - 1

    if test_story is not None:
        try:
            stories.remove(test_story)
        except ValueError as e:
            log.critical(f"test_story: {test_story} is not in the pool of all stories.")
            raise e

    rng = np.random.default_rng(seed=seed)

    # data dicts
    X_data_dict = dict()
    y_data_dict = dict()
    # result arrays
    scores_list = list()
    weights_list = list()
    best_alphas_list = list()
    for repeat in range(n_repeats):
        log.info(f"Repeat {repeat}")

        # 1. choose stories to sample for this repeat
        if test_story is None:
            curr_all_stories: list[str] = rng.choice(
                stories, size=n_train_stories + 1, replace=False
            ).tolist()
            curr_train_stories = curr_all_stories[:-1]
            curr_test_stories = curr_all_stories[-1:]
        else:
            curr_train_stories = rng.choice(
                stories, size=n_train_stories, replace=False
            ).tolist()
            curr_test_stories = [test_story]
            curr_all_stories = [*curr_train_stories, *curr_test_stories]

        # 2. load data for stories
        stories_to_load = list(
            set(curr_all_stories).difference(set(X_data_dict.keys()))
        )
        if len(stories_to_load) > 0:
            log.info(f"{repeat} | Loading data")
            X_data_dict_new, y_data_dict_new = load_data_dict(
                stories_to_load,
                subject,
                predictor,
                interpolation,
                shuffle,
                tr_len,
                n_delays,
                use_cache,
            )
            X_data_dict.update(X_data_dict_new)
            y_data_dict.update(y_data_dict_new)

        # 3. run regression
        log.info(f"{repeat} | Running Regression")
        if ridge_implementation == "ridge_huth":
            scores, weights, best_alphas = ridge_regression_huth(
                train_stories=curr_train_stories,
                test_stories=curr_test_stories,
                X_data_dict=X_data_dict,
                y_data_dict=y_data_dict,
                score_fct=pearsonr,  # type: ignore
            )
        else:
            scores, weights, best_alphas = ridge_regression(
                train_stories=curr_train_stories,
                test_stories=curr_test_stories,
                X_data_dict=X_data_dict,
                y_data_dict=y_data_dict,
                score_fct=pearsonr,  # type: ignore
            )
        log.info(f"{repeat} | Mean corr: {scores.mean()}")
        log.info(f"{repeat} | Max corr : {scores.max()}")

        if not keep_train_stories_in_mem:
            del X_data_dict
            del y_data_dict
            X_data_dict = dict()
            y_data_dict = dict()

        # 4. append results
        scores_list.append(scores)
        weights_list.append(weights)
        best_alphas_list.append(best_alphas)

    # aggregate scores
    mean_scores = np.mean(scores_list, axis=0)

    return mean_scores, scores_list, weights_list, best_alphas_list


def do_regression(
    strategy: str = "loocv",
    predictor: str = "embeddings",
    stories: list[str] = STORIES,
    n_train_stories: Optional[int] = None,
    test_story: Optional[str] = None,
    n_repeats: int = 5,
    subject: str = "UTS02",
    tr_len: float = 2.0,
    n_delays: int = 4,
    interpolation: str = "lanczos",
    ridge_implementation: str = "ridge_huth",
    use_cache: bool = True,
    shuffle: bool = False,
    seed: Optional[int] = 123,
    show_results: bool = True,
    keep_train_stories_in_mem: bool = True,
) -> tuple[
    np.ndarray, list[np.ndarray], list[np.ndarray], list[Union[float, np.ndarray]]
]:
    """Runs regression.

    Parameters
    ----------
    strategy : {"loocv", "simple"}, default="loocv"
        `loocv` uses leave-one-out cross-validation for n_stories. The stories are
        determined by the order of the `stories` parameter or its default value in
        `config.yaml`.
        `simple` computes the regression for a train/test split containing n_stories
        within each repeat.
        Stories are sampled randomly for each repeat.
    predictor : {"envelope", "embeddings", "embeddings_huth"}
        Which predictor to use for the regression. "envelope": audio envelope that
        participants heard. "embeddings": embeddings computed from the words that
        participants heard. "embeddings_huth": embeddings which were used in the
        huth paper (https://www.nature.com/articles/s41597-023-02437-z)
    stories: list[str], default=STORIES
        Pool of stories, the default is determined by the config.yaml
    n_train_stories: int, optional
        The amount of training stories sampled from `stories`, if this is `None` will
        use all except one story.
    test_story : str or `None`, default=`None`
        Only used if `strategy="simple"`. The story on which the regression models will
        be tested.
        If `None`, the test story will be randomly selected from the pool of stories.
    n_repeats : int, default=5
        Only used if `strategy="simple"`. Determines how often regression is repeated
        on a different train/test set.
    subject : {"UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06", "UTS07", "UTS08"},
        default="UTS02"
        Subject identifier
    tr_len: float, default=2.0
        Length of tr-windows used to sample fMRI data.
    n_delays: int, default=4
        How many delays are used to model the HRF, which is modeled by adding
        a shifted set of duplicated features for each delay. `n_delays=5` implies
        that the the features of the stimulus are shifted concatinated 5 times
        to training/testing data.
    interpolation: {"lanczos", "average"}, default="lanczos"
        Whether to use lanczos interpolation or just average the words within a TR.
        Only applies to the 'embeddings' predictor.
    ridge_implementation: {"ridgeCV", "ridge_huth"}, default="ridge_huth"
        Which implementation of ridge regression to use. "ridgeCV" uses the RidgeCV
        function from sklearn.
        "ridge_huth" uses the implementation from the Huth lab codebase which applies
        SVD to the data matrix and computes correlation scores with bootstrapping.
    use_cache: bool, default=True
        Whether the cache is used for `envelope` features.
    shuffle: bool, default=False
        Whether to shuffle the predictors (features).
    seed: int | None, default=123
        Seed determining sampling of stories
    show_results: bool, default=True
        Create a plot showing the results in pycortex.
    keep_train_stories_in_mem: bool, default=True
        Whether stories are kept in memory after first loading. Unless when using all
        stories turning this off will reduce the memory footprint, but increase the
        time is spent loading data. Only works if `strategy='simple'`.


    Returns
    -------
    mean_scores : np.ndarray
        The mean prediction scores for each repeat/split.
    scores_list : list of np.ndarray
        The scores for each repeat/split.
    weights : list of np.ndarray
        The regression weights for each repeat/split.
    alphas : list of float or list of np.ndarray
        The best alphas for each voxel, for each repeat/split.
    """

    if n_train_stories is None:
        n_train_stories = len(stories) - 1

    if strategy == "loocv":
        mean_scores, all_scores, all_weights, best_alphas = do_loocv_regression(
            predictor=predictor,
            stories=stories,
            n_train_stories=n_train_stories,
            subject=subject,
            tr_len=tr_len,
            n_delays=n_delays,
            interpolation=interpolation,
            ridge_implementation=ridge_implementation,
            use_cache=use_cache,
            shuffle=shuffle,
        )
    elif strategy == "simple":
        mean_scores, all_scores, all_weights, best_alphas = do_simple_regression(
            predictor=predictor,
            stories=stories,
            n_train_stories=n_train_stories,
            test_story=test_story,
            subject=subject,
            tr_len=tr_len,
            n_delays=n_delays,
            interpolation=interpolation,
            ridge_implementation=ridge_implementation,
            use_cache=use_cache,
            shuffle=shuffle,
            n_repeats=n_repeats,
            seed=seed,
            keep_train_stories_in_mem=keep_train_stories_in_mem,
        )
    else:
        raise ValueError(f"Invalid regression strategy: {strategy}")

    if show_results:
        plt.plot(mean_scores)
        plt.show()
    log.info(f"Mean correlation (averages across splits) (r): {mean_scores.mean()}")
    log.info(f"Max  correlation (averaged across splits) (r): {mean_scores.max()}")

    if show_results:
        vol_data = cortex.Volume(
            mean_scores, subject, f"{subject}_auto", vmin=0, vmax=0.5, cmap="inferno"
        )

        cortex.quickshow(vol_data)
        plt.title(f"{subject} {predictor} performance.")
        plt.show()
        plt.savefig(
            os.path.join("data", f"{predictor}_{subject}_{n_train_stories}.png")
        )
        # save the plot
        _ = cortex.quickflat.make_png(
            os.path.join("data", f"{predictor}_{strategy}_{subject}.png"),
            vol_data,
            recache=False,
        )
        # without print statement the plot does not show up.
        print("Done")
    return mean_scores, all_scores, all_weights, best_alphas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A program to run a ")
    parser.add_argument(
        "strategy",
        choices=["simple", "loocv"],
        help="The predictor",
    )
    parser.add_argument(
        "predictor",
        choices=["embeddings", "envelope", "embeddings_huth"],
        help="The predictor",
    )
    parser.add_argument(
        "--n_train_stories",
        default=5,
        type=int,
        help="How many stories are used. Averages LOO results.",
    )
    parser.add_argument(
        "--n_repeats",
        default=5,
        type=int,
        help=(
            "Only used if `strategy='simple'`. Determines how often regression"
            " is repeated on a different train/test set."
        ),
    )
    parser.add_argument(
        "--subject",
        default="UTS02",
        type=str,
        help="Subject identifier in the downloaded dataset.",
    )
    parser.add_argument(
        "--not_use_cache",
        action="store_true",
        help="Disable cache for feature commputation.",
    )
    parser.add_argument(
        "--interpolation_method",
        default="lanczos",
        choices=["lanczos", "average"],
        help="Interpolation method for embeddings.",
    )
    parser.add_argument(
        "--ridge_implementation",
        default="ridge_huth",
        choices=["ridgeCV", "ridge_huth"],
        help="Which implementation of ridge regression to use.",
    )
    parser.add_argument(
        "--n_delays",
        default=4,
        type=int,
        help="Delays to model the HRF.",
    )

    args = parser.parse_args()
    do_regression(
        strategy=args.strategy,
        predictor=args.predictor,
        n_train_stories=args.n_train_stories,
        n_repeats=args.n_repeats,
        subject=args.subject,
        interpolation=args.interpolation_method,
        ridge_implementation=args.ridge_implementation,
        use_cache=not args.not_use_cache,
        n_delays=args.n_delays,
    )
