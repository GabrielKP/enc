import itertools as itools
import os
import random
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Union

import cortex
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sem
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from encoders.features import load_data_dict
from encoders.utils import counter, get_logger, load_config

log = get_logger(__name__)
cfg = load_config()

DATADIR = Path(cfg["DATA_DIR"])
STORIES = cfg["STORIES"]
WAV_DIR = "stimuli"


def zs(x: np.ndarray) -> np.ndarray:
    """Returns the z-score of the input array. Z-scores along the first dimension for
    n-dimensional arrays by default.

    Parameters
    ----------
    x : np.ndarray
        Input array. Can be 1D or n-dimensional.

    Returns
    -------
    z : np.ndarray
        Z-score of x.
    """
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)


def z_score(data: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """
    Return `data` z-scored by given `means` and standard deviations `stds`. Useful
    to z-score after train/test splits with the same mean in cross-validation settings
    to prevent data leakage.

    Parameters
    ----------
    data : np.ndarray
        shape = (n_samples, n_features) or (n_samples,)
    means : np.ndarray
        shape = (n_features,) or (1, n_features)
    stds : np.ndarray
        shape = (n_features,) or (1, n_features)

    Returns
    -------
    np.ndarray
       Normalized data, shape = (n_samples, n_features) or (1, n_features)

    """
    return (data - means) / (stds + 1e-6)


def pearsonr(x1: np.ndarray, x2: np.ndarray) -> Union[float, np.ndarray]:
    """Returns the pearson correlation between two vectors or two matrices of the same
    shape (in which case the correlation is computed for each pair of column vectors).

    Parameters
    ----------
    x1 : np.ndarray
        shape = (n_samples,) or (n_samples, n_targets)
    x2 : np.ndarray
        shape = (n_samples,) or (n_samples, n_targets), same shape as x1

    Returns
    -------
    corr: float or np.ndarray
        Pearson correlation between x1 and x2. If x1 and x2 are matrices, returns an
        array of correlations with shape (n_targets,)
    """
    return np.mean(zs(x1) * zs(x2), axis=0)


def pearsonr_scorer(
    estimator, X: np.ndarray, y: np.ndarray
) -> Union[float, np.ndarray]:
    """Scorer function for RidgeCV that computes the Pearson correlation between the
    predicted and true values.

    Parameters
    ----------
    estimator : object
        A trained scikit-learn estimator with a `predict` method.
    X : np.ndarray
        The input data used for prediction.
    y : np.ndarray
        The true target values.

    Returns:
    --------
    float
        The Pearson correlation coefficient between the predicted and true values.
    """
    y_predict = estimator.predict(X)
    return pearsonr(y, y_predict)


def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array

    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))

    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d * mtx.T).T
    else:
        return d * mtx


def ridge_regression(
    train_stories: list[str],
    test_stories: list[str],
    X_data_dict: dict[str, np.ndarray],
    y_data_dict: dict[str, np.ndarray],
    score_fct: Callable[[np.ndarray, np.ndarray], np.ndarray],
    alphas: Optional[np.ndarray] = np.logspace(1, 3, 10),
) -> tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """Runs CVRidgeRegression on given train stories, returns scores for
    test story, regression weights, and the best alphas.

    Parameters
    ----------
    train_stories: list of str
        List of training stories. Stories must be in X_data_dict.
    test_stories: list of str
        List of testing stories. Stories must be in X_data_dict.
    X_data_dict : dict[str, np.ndarray]
        Dict with story to X_data (features) pairs.
    y_data_dict : dict[str, np.ndarray]
        Dict with story to y_data (fMRI data) pairs.
    score_fct : fct(np.ndarray, np.ndarray) -> np.ndarray
        A function taking y_test (shape = (number_trs, n_voxels))
        and y_predict (same shape as y_test) and returning an
        array with an entry for each voxel (shape = (n_voxels)).
    alphas : np.ndarray or `None`, default = `np.logspace(1, 3, 10)`
        Array of alpha values to optimize over. If `None`, will choose
        default value.

    Returns
    -------
    scores : np.ndarray
        The prediction scores for the test stories.
    weights : np.ndarray
        The regression weights.
    best_alphas : float | np.ndarray
        The best alphas for each voxel.
    """
    if alphas is None:
        alphas = np.logspace(1, 3, 10)

    X_train_list = [X_data_dict[story] for story in train_stories]
    y_train_list = [y_data_dict[story] for story in train_stories]
    X_test_list = [X_data_dict[story] for story in test_stories]
    y_test_list = [y_data_dict[story] for story in test_stories]

    X_train_unnormalized = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test_unnormalized = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    X_means = X_train_unnormalized.mean(axis=0)
    X_stds = X_train_unnormalized.std(axis=0)

    X_train = z_score(X_train_unnormalized, X_means, X_stds)
    X_test = z_score(X_test_unnormalized, X_means, X_stds)

    clf = RidgeCV(alphas=alphas, alpha_per_target=True, scoring=pearsonr_scorer)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    scores = score_fct(y_test, y_predict)

    return scores, clf.coef_, clf.alpha_


def ridge_regression_huth(
    train_stories: list[str],
    test_stories: list[str],
    X_data_dict: dict[str, np.ndarray],
    y_data_dict: dict[str, np.ndarray],
    score_fct: Callable[[np.ndarray, np.ndarray], np.ndarray],
    alphas: Optional[np.ndarray] = np.logspace(1, 3, 10),
    nboots: int = 50,
    chunklen: int = 40,
    nchunks: int = 125,
    singcutoff: float = 1e-10,
    single_alpha: bool = False,
) -> tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """Runs CVRidgeRegression on given train stories, returns scores for
    test story, regression weights, and the best alphas. Instead of using RidgeCV from
    sklearn, uses the Huth lab's implementation
    (https://github.com/HuthLab/deep-fMRI-dataset/blob/master/encoding/ridge_utils/ridge.py).

    Parameters
    ----------
    train_stories: list of str
        List of training stories. Stories must be in X_data_dict.
    test_stories: list of str
        List of testing stories. Stories must be in X_data_dict.
    X_data_dict : dict[str, np.ndarray]
        Dict with story to X_data (features) pairs.
    y_data_dict : dict[str, np.ndarray]
        Dict with story to y_data (fMRI data) pairs.
    score_fct : fct(np.ndarray, np.ndarray) -> np.ndarray
        A function taking y_test (shape = (number_trs, n_voxels))
        and y_predict (same shape as y_test) and returning an
        array with an entry for each voxel (shape = (n_voxels)).
    alphas : np.ndarray or `None`, default = `np.logspace(1, 3, 10)`
        Array of alpha values to optimize over. If `None`, will choose
        default value.

    Returns
    -------
    scores : np.ndarray
        The prediction scores for the test stories.
    weights : np.ndarray
        The regression weights.
    best_alphas : float | np.ndarray
        The best alphas for each voxel.
    """
    if alphas is None:
        alphas = np.logspace(1, 3, 10)

    X_train_list = [X_data_dict[story] for story in train_stories]
    y_train_list = [y_data_dict[story] for story in train_stories]
    X_test_list = [X_data_dict[story] for story in test_stories]
    y_test_list = [y_data_dict[story] for story in test_stories]

    X_train_unnormalized = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test_unnormalized = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    X_means = X_train_unnormalized.mean(axis=0)
    X_stds = X_train_unnormalized.std(axis=0)

    X_train = z_score(X_train_unnormalized, X_means, X_stds)
    X_test = z_score(X_test_unnormalized, X_means, X_stds)

    wt, _, bestalphas, _, _ = bootstrap_ridge(
        X_train,
        y_train,
        X_test,
        y_test,
        alphas,
        nboots,
        chunklen,
        nchunks,
        singcutoff=singcutoff,
        single_alpha=single_alpha,
    )
    scores = score_fct(np.dot(X_test, wt), y_test)

    return scores, wt, bestalphas


def crossval_loocv(
    predictor: str = "embeddings",
    stories: Union[list[str], None] = None,
    n_train_stories: Optional[int] = None,
    subject: str = "UTS02",
    tr_len: float = 2.0,
    n_delays: int = 4,
    interpolation: str = "lanczos",
    ridge_implementation: str = "ridge_huth",
    alphas: Optional[np.ndarray] = None,
    use_cache: bool = True,
    shuffle: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    # 1. choose stories
    # the order of stories is determined in config.yaml or via the argument.
    if stories is None:
        stories = STORIES.copy()
        if not isinstance(stories, list):
            raise ValueError(f"Config parameter invalid: STORIES: {stories}")

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

    return np.array(scores_list), weights_list, best_alphas_list


def crossval_simple(
    predictor: str = "embeddings",
    stories: Union[list[str], None] = None,
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
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
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
        If `cross_validation="simple"`, determines how often regression is
        repeated on a different train/test set.
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

    if stories is None:
        stories = STORIES.copy()
        if not isinstance(stories, list):
            raise ValueError(f"Config parameter invalid: STORIES: {stories}")

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
            log.info("Loading data")
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
        log.info(
            f"Running regression | n_train_stories: {n_train_stories}"
            + f" | implementation: {ridge_implementation}"
        )
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
        log.info(f"Mean corr: {scores.mean()}")
        log.info(f"Max corr : {scores.max()}")

        if not keep_train_stories_in_mem:
            del X_data_dict
            del y_data_dict
            X_data_dict = dict()
            y_data_dict = dict()

        # 4. append results
        scores_list.append(scores)
        weights_list.append(weights)
        best_alphas_list.append(best_alphas)

    return np.array(scores_list), weights_list, best_alphas_list


def do_regression(
    cross_validation: str = "loocv",
    predictor: str = "embeddings",
    stories: Union[list[str], None] = None,
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
    tuple[np.ndarray, list[np.ndarray]],
    tuple[np.ndarray, list[np.ndarray], list[np.ndarray]],
]:
    """Runs regression.

    Parameters
    ----------
    cross_validation : {"loocv", "simple"}, default="loocv"
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
        Only used if `cross_validation="simple"`. The story on which
        the regression models will be tested.
        If `None`, the test story will be randomly selected from the pool of stories.
    n_repeats : int, default=5
        Only used if `cross_validation="simple"`. Determines how often
        regression is repeated on a different train/test set.
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
        time is spent loading data. Only works if `cross_validation='simple'`.


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

    if stories is None:
        stories = STORIES.copy()
        if not isinstance(stories, list):
            raise ValueError(f"Config parameter invalid: STORIES: {stories}")

    if n_train_stories is None:
        n_train_stories = len(stories) - 1

    if cross_validation == "loocv":
        all_scores, all_weights, best_alphas = crossval_loocv(
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
    elif cross_validation == "simple":
        all_scores, all_weights, best_alphas = crossval_simple(
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
        raise ValueError(f"Invalid cross validation method: {cross_validation}")

    # aggregate scores
    mean_scores = np.mean(all_scores, axis=0)
    sem_scores = sem([np.mean(fold_data, axis=0) for fold_data in all_scores], axis=0)

    log.info(f"Mean correlation (averages across splits) (r): {mean_scores.mean()}")
    log.info(f"Max  correlation (averaged across splits) (r): {mean_scores.max()}")

    return (mean_scores, sem_scores), (all_scores, all_weights, best_alphas)


######### Below is the original code from the Huth lab's implementation #########


def ridge_corr(
    Rstim,
    Pstim,
    Rresp,
    Presp,
    alphas,
    normalpha=False,
    dtype=np.single,
    corrmin=0.2,
    singcutoff=1e-10,
    use_corr=True,
    logger=get_logger("ridge_corr"),
):
    """Uses ridge regression to find a linear transformation of [Rstim] that
    approximates [Rresp]. Then tests by comparing the transformation of [Pstim] to
    [Presp]. This procedure is repeated for each regularization parameter alpha in
    [alphas]. The correlation between each prediction and each response for each alpha
    is returned. Note that the regression weights are NOT returned.

    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be
        Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored
        across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons,
        what-have-you). Each response should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    alphas : list or array_like, shape (A,)
        Ridge parameters to be tested. Should probably be log-spaced.
        np.logspace(0, 3, 20) works well.
    normalpha : boolean
        Whether ridge parameters should be normalized by the Frobenius norm of Rstim.
        Good for comparing models with different numbers of parameters.
    dtype : np.dtype
        All data will be cast as this dtype for computation. np.single is used by
        default for memory efficiency.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses
        with correlation greater than corrmin minus the number of responses with
        correlation less than negative corrmin will be printed. For long-running
        regressions this vague metric of non-centered skewness can give you a rough
        sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition
        (SVD) of the stimulus Rstim. If Rstim is not full rank, some singular values
        will be approximately equal to zero and the corresponding singular vectors will
        be noise. These singular values/vectors should be removed both for speed (the
        fewer multiplications the better!) and accuracy. Any singular values less than
        singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If
        False, this function will instead use variance explained (R-squared) as its
        metric of model fit. For ridge regression this can make a big difference --
        highly regularized solutions will have very small norms and will thus explain
        very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.

    Returns
    -------
    Rcorrs : array_like, shape (A, M)
        The correlation between each predicted response and each column of Presp for
        each alpha.

    """
    ## Calculate SVD of stimulus matrix
    logger.info("Doing SVD...")
    try:
        U, S, Vh = np.linalg.svd(Rstim, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd  # type: ignore

        U, S, Vh = svd_dgesvd(Rstim, full_matrices=False)

    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S > singcutoff)
    nbad = origsize - ngoodS
    U = U[:, :ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.info(
        "Dropped %d tiny singular values.. (U is now %s)" % (nbad, str(U.shape))
    )

    ## Normalize alpha by the Frobenius norm
    # frob = np.sqrt((S**2).sum()) ## Frobenius!
    frob = S[0]
    # frob = S.sum()
    logger.info("Training stimulus has Frobenius norm: %0.03f" % frob)
    if normalpha:
        nalphas = alphas * frob
    else:
        nalphas = alphas

    ## Precompute some products for speed
    UR = np.dot(U.T, Rresp)  ## Precompute this matrix product for speed
    PVh = np.dot(Pstim, Vh.T)  ## Precompute this matrix product for speed

    # Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test
    # response norms
    zPresp = zs(Presp)
    Prespvar = Presp.var(0)
    Rcorrs = []  ## Holds training correlations for each alpha
    for na, a in zip(nalphas, alphas):
        # D = np.diag(S/(S**2+a**2)) ## Reweight singular vectors by the ridge parameter
        D = S / (
            S**2 + na**2
        )  ## Reweight singular vectors by the (normalized?) ridge parameter

        pred = np.dot(
            mult_diag(D, PVh, left=False), UR
        )  ## Best (1.75 seconds to prediction in test)
        # pred = np.dot(mult_diag(D, np.dot(Pstim, Vh.T), left=False), UR) ## Better
        # (2.0 seconds to prediction in test)

        # pvhd = reduce(np.dot, [Pstim, Vh.T, D]) ## Pretty good (2.4 seconds to
        # prediction in test)
        # pred = np.dot(pvhd, UR)

        # wt = reduce(np.dot, [Vh.T, D, UR]).astype(dtype) ## Bad (14.2 seconds to
        # prediction in test)
        # wt = reduce(np.dot, [Vh.T, D, U.T, Rresp]).astype(dtype) ## Worst
        # pred = np.dot(Pstim, wt) ## Predict test responses

        if use_corr:
            # prednorms = np.apply_along_axis(np.linalg.norm, 0, pred) ## Compute
            # predicted test response norms
            # Rcorr = np.array(
            #    [np.corrcoef(Presp[:,ii], pred[:,ii].ravel())[0,1]
            #     for ii in range(Presp.shape[1])]) ## Slowly compute correlations
            # Rcorr = np.array(np.sum(np.multiply(Presp, pred), 0)).squeeze()/
            #            (prednorms*Prespnorms) ## Efficiently compute correlations
            Rcorr = (zPresp * zs(pred)).mean(0)
        else:
            ## Compute variance explained
            resvar = (Presp - pred).var(0)
            Rcorr = np.clip(1 - (resvar / Prespvar), 0, 1)

        Rcorr[np.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)

        log_template = (
            "Training: alpha=%0.3f, mean corr=%0.5f,"
            " max corr=%0.5f, over-under(%0.2f)=%d"
        )
        log_msg = log_template % (
            a,
            np.mean(Rcorr),
            np.max(Rcorr),
            corrmin,
            (Rcorr > corrmin).sum() - (-Rcorr > corrmin).sum(),
        )
        if logger is not None:
            logger.info(log_msg)
        else:
            print(log_msg)

    return Rcorrs


def bootstrap_ridge(
    Rstim,
    Rresp,
    Pstim,
    Presp,
    alphas,
    nboots,
    chunklen,
    nchunks,
    dtype=np.single,
    corrmin=0.2,
    joined=None,
    singcutoff=1e-10,
    normalpha=False,
    single_alpha=False,
    use_corr=True,
    logger=get_logger("ridge_corr"),
):
    """Uses ridge regression with a bootstrapped held-out set to get optimal alpha
    values for each response. [nchunks] random chunks of length [chunklen] will be taken
    from [Rstim] and [Rresp] for each regression run.  [nboots] total regression runs
    will be performed.  The best alpha value for each response will be averaged across
    the bootstraps to estimate the best alpha for that response.

    If [joined] is given, it should be a list of lists where the STRFs for all the
    voxels in each sublist will be given the same regularization parameter (the one
    that is the best on average).

    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be
        Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M different responses (voxels,
        neurons, what-have-you). Each response should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be
        Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M different responses. Each response
        should be Z-scored across
        time.
    alphas : list or array_like, shape (A,)
        Ridge parameters that will be tested. Should probably be log-spaced.
        np.logspace(0, 3, 20) works well.
    nboots : int
        The number of bootstrap samples to run. 15 to 30 works well.
    chunklen : int
        On each sample, the training data is broken into chunks of this length. This
        should be a few times
        longer than your delay/STRF. e.g. for a STRF with 3 delays, I use chunks of
        length 10.
    nchunks : int
        The number of training chunks held out to test ridge parameters for each
        bootstrap sample. The product of nchunks and chunklen is the total number of
        training samples held out for each sample, and this product should be about 20
        percent of the total length of the training data.
    dtype : np.dtype
        All data will be cast as this dtype for computation. np.single is used by
        default for memory efficiency, as using np.double will thrash most machines on a
        big problem. If you want to do regression on complex variables, this should be
        changed to np.complex128.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested for each bootstrap
        sample, the number of responses with correlation greater than this value will be
        printed. For long-running regressions this can give a rough sense of how well
        the model works before it's done.
    joined : None or list of array_like indices
        If you want the STRFs for two (or more) responses to be directly comparable, you
        need to ensure that the regularization parameter that they use is the same. To
        do that, supply a list of the response sets that should use the same ridge
        parameter here. For example, if you have four responses, joined could be
        [np.array([0,1]), np.array([2,3])], in which case responses 0 and 1 will use the
        same ridge parameter (which will be parameter that is best on average for those
        two), and likewise for responses 2 and 3.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition
        (SVD) of the stimulus Rstim. If Rstim is not full rank, some singular values
        will be approximately equal to zero and the corresponding singular vectors will
        be noise. These singular values/vectors should be removed both for speed (the
        fewer multiplications the better!) and accuracy. Any singular values less than
        singcutoff will be removed.
    normalpha : boolean
        Whether ridge parameters (alphas) should be normalized by the Frobenius norm of
        Rstim. Good for rigorously comparing models with different numbers of
        parameters.
    single_alpha : boolean
        Whether to use a single alpha for all responses. Good
        foridentification/decoding.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If
        False, this function will instead use variance explained (R-squared) as its
        metric of model fit. For ridge regression this can make a big difference --
        highly regularized solutions will have very small norms and will thus explain
        very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.

    Returns
    -------
    wt : array_like, shape (N, M)
        Regression weights for N features and M responses.
    corrs : array_like, shape (M,)
        Validation set correlations. Predicted responses for the validation set are
        obtained using the regression weights: pred = np.dot(Pstim, wt), and then the
        correlation between each predicted response and each column in Presp is found.
    alphas : array_like, shape (M,)
        The regularization coefficient (alpha) selected for each voxel using bootstrap
        cross-validation.
    bootstrap_corrs : array_like, shape (A, M, B)
        Correlation between predicted and actual responses on randomly held out portions
        of the training set, for each of A alphas, M voxels, and B bootstrap samples.
    valinds : array_like, shape (TH, B)
        The indices of the training data that were used as "validation" for each
        bootstrap sample.
    """
    nresp, nvox = Rresp.shape
    valinds = []  ## Will hold the indices into the validation data for each bootstrap

    Rcmats = []
    for bi in counter(range(nboots), countevery=1, total=nboots):
        logger.info("Selecting held-out test set..")
        allinds = range(nresp)
        indchunks = list(zip(*[iter(allinds)] * chunklen))
        random.shuffle(indchunks)
        heldinds = list(itools.chain(*indchunks[:nchunks]))
        notheldinds = list(set(allinds) - set(heldinds))
        valinds.append(heldinds)

        RRstim = Rstim[notheldinds, :]
        PRstim = Rstim[heldinds, :]
        RRresp = Rresp[notheldinds, :]
        PRresp = Rresp[heldinds, :]

        ## Run ridge regression using this test set
        Rcmat = ridge_corr(
            RRstim,
            PRstim,
            RRresp,
            PRresp,
            alphas,
            dtype=dtype,
            corrmin=corrmin,
            singcutoff=singcutoff,
            normalpha=normalpha,
            use_corr=use_corr,
        )

        Rcmats.append(Rcmat)

    ## Find weights for each voxel
    try:
        U, S, Vh = np.linalg.svd(Rstim, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd  # type: ignore

        U, S, Vh = svd_dgesvd(Rstim, full_matrices=False)

    ## Normalize alpha by the Frobenius norm
    # frob = np.sqrt((S**2).sum()) ## Frobenius!
    frob = S[0]
    # frob = S.sum()
    logger.info("Total training stimulus has Frobenius norm: %0.03f" % frob)
    if normalpha:
        nalphas = alphas * frob
    else:
        nalphas = alphas

    allRcorrs = np.dstack(Rcmats)
    if not single_alpha:
        logger.info("Finding best alpha for each response..")
        if joined is None:
            ## Find best alpha for each voxel
            meanbootcorrs = allRcorrs.mean(2)
            bestalphainds = np.argmax(meanbootcorrs, 0)
            valphas = nalphas[bestalphainds]
        else:
            ## Find best alpha for each group of voxels
            valphas = np.zeros((nvox,))
            for jl in joined:
                jcorrs = (
                    allRcorrs[:, jl, :].mean(1).mean(1)
                )  ## Mean across voxels in the set, then mean across bootstraps
                bestalpha = np.argmax(jcorrs)
                valphas[jl] = nalphas[bestalpha]
    else:
        logger.info("Finding single best alpha..")
        meanbootcorr = allRcorrs.mean(2).mean(1)
        bestalphaind = np.argmax(meanbootcorr)
        bestalpha = alphas[bestalphaind]
        valphas = np.array([bestalpha] * nvox)
        logger.info("Best alpha = %0.3f" % bestalpha)

    logger.info("Computing weights for each response using entire training set..")
    UR = np.dot(U.T, np.nan_to_num(Rresp))
    pred = np.zeros(Presp.shape)
    wt = np.zeros((Rstim.shape[1], Rresp.shape[1]))
    for ai, alpha in enumerate(nalphas):
        selvox = np.nonzero(valphas == alpha)[0]
        awt = reduce(np.dot, [Vh.T, np.diag(S / (S**2 + alpha**2)), UR[:, selvox]])
        pred[:, selvox] = np.dot(Pstim, awt)
        wt[:, selvox] = awt

    ## Find test correlations
    nnpred = np.nan_to_num(pred)
    corrs = np.nan_to_num(
        np.array(
            [
                np.corrcoef(Presp[:, ii], nnpred[:, ii].ravel())[0, 1]
                for ii in range(Presp.shape[1])
            ]
        )
    )

    return wt, corrs, valphas, allRcorrs, valinds
