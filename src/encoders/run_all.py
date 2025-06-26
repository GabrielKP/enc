import argparse
import json
import os
from collections import defaultdict
from functools import partial
from itertools import product
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.stats import sem

from encoders.regression import crossval_loocv, crossval_simple
from encoders.utils import (
    check_make_dirs,
    create_run_folder_name,
    get_logger,
    load_config,
)

log = get_logger(__name__)

cfg = load_config()
RUNS_DIR = cfg["RUNS_DIR"]
TR_LEN = cfg["TR_LEN"]


def run_all(
    cross_validation: str = "simple",
    subject: Union[str, list[str]] = "UTS02",
    feature: Union[str, list[str]] = "eng1000",
    n_train_stories: Union[int, list[int]] = [1, 3, 5],
    test_story: Optional[str] = "wheretheressmoke",
    n_repeats: int = 5,
    ndelays: int = 5,
    interpolation: str = "lanczos",
    ridge_implementation: str = "ridge_huth",
    do_shuffle: bool = False,
    use_cache: bool = True,
    keep_train_stories_in_mem: bool = True,
    seed: Optional[int] = 123,
    alphas: np.ndarray = np.logspace(1, 3, 10),
    nboots: int = 50,
    chunklen: int = 40,
    nchunks: int = 125,
    singcutoff: float = 1e-10,
    single_alpha: bool = False,
    use_corr: bool = True,
    run_folder_name: str = "",
):
    """Runs encoding models n_repeat times and saves results data/runs to disk.

    The following outputs are saved:
    0. params.json : the parameters for the run
        `data/runs/date-id/params.json`
    1. scores_mean.npy : the mean scores across folds
        `data/runs/date-id/predictor/subject/n_training_stories/shuffle/scores_mean.npy`
    2. scores.npy : the score for each separate fold
        `data/runs/date-id/predictor/subject/n_training_stories/shuffle/fold/scores.npy`
    3. weights.npy : the model weights for each separate fold
        `data/runs/date-id/predictor/subject/n_training_stories/shuffle/fold/weights.npy`
    4. best_alphas.npy : the best alphas for each voxel in that fold
        `data/runs/date-id/predictor/subject/n_training_stories/shuffle/fold/best_alphas.npy`


    Parameters
    ----------
    cross_validation : {"loocv", "simple"}, default="simple"
        `loocv` uses leave-one-out cross-validation for n_stories. The stories are
         determined by the order of the `stories` parameter or its default value in
         `config.yaml`.
        `simple` computes the regression for a train/test split containing n_stories
         within each repeat.
        Stories are sampled randomly for each repeat.
    subject : str or list of str, default="UTS02"
        Subject identifier.
        Can be one or a list of : {`"all"`, `"UTS01"`, `"UTS02"`, `"UTS03"`, `"UTS04"`,
         `"UTS05"`, `"UTS06"`, `"UTS07"`, `"UTS08"`}
    feature : {"all", "envelope", "eng1000"}, default="eng1000"
        Which predictor to run.
        `all` will run separate encoding models for both predictors (default).
        `envelope` will run the encoding model with the audio envelope as predictor.
        `eng1000` will run the encoding model with the word embeddings of the stories
         as predictor.
    n_train_stories : int or list of int
        Number o of training stories for the encoding model. If a list is given, the
         encoding model will be fitted with each number separately.
    test_story : str or None, default="wheretheressmoke"
        The story to use as the test set. If `None`, it will be sampled randomly.
    ndelays : int, default=5
        By how many TR's features are delayed to model the HRF. For `ndelays=5`, the
         features of the predictor are shifted by one TR and concatinated to themselves
         for five times.
    interpolation : {"lanczos", "average"}, default="lanczos"
        Whether to use lanczos interpolation or just average the words within a TR.
        Only applies if `predictor=eng1000`.
    ridge_implementation: {"ridgeCV", "ridge_huth"}, default="ridge_huth"
        Which ridge regression implementation to use.
        `ridgeCV` will use scikit-learn's RidgeCV.
        `ridge_huth` will use the ridge regression implementation from Lebel et al.
    do_shuffle: book, default=False
        Whether or not to run model fits with predictors shuffled (as a control).
        A separate subfolder ('shuffled') will be created in the run folder with
        these results.
    use_cache: bool, default=True
        Whether features are cached and reused.
        Only applies if `predictor=envelope`.
    keep_train_stories_in_mem: bool, default=True
        Whether stories are kept in memory after first loading. Unless when using all
        stories turning this off will reduce the memory footprint, but increase the
        time is spent loading data. Only works if `cross_validation='simple'`.
    seed: int | None, default=123
        Seed determining sampling of stories
    alphas : np.ndarray
        Array of alpha values to optimize over.
    nboots : int
        The number of bootstrap samples to run. 15 to 30 works well.
        Only active for ridge_huth="ridge_huth".
    chunklen : int
        On each sample, the training data is broken into chunks of this length. This
        should be a few times longer than your delay/STRF. e.g. for a STRF with 3
        delays, I use chunks of length 10.
        Only active for ridge_huth="ridge_huth".
    nchunks : int
        The number of training chunks held out to test ridge parameters for each
        bootstrap sample. The product of nchunks and chunklen is the total number of
        training samples held out for each sample, and this product should be about 20
        percent of the total length of the training data.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition
        (SVD) of the stimulus Rstim. If Rstim is not full rank, some singular values
        will be approximately equal to zero and the corresponding singular vectors will
        be noise. These singular values/vectors should be removed both for speed (the
        fewer multiplications the better!) and accuracy. Any singular values less than
        singcutoff will be removed.
        Only active for ridge_huth="ridge_huth".
    single_alpha : boolean
        Whether to use a single alpha for all responses. Good
        foridentification/decoding.
        Only active for ridge_huth="ridge_huth".
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If
        False, this function will instead use variance explained (R-squared) as its
        metric of model fit. For ridge regression this can make a big difference --
        highly regularized solutions will have very small norms and will thus explain
        very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
        Only active for ridge_huth="ridge_huth".
    run_folder_name: str, optional
        The name of the folder in the runs directory (as specificed in
        `encoders.utils.load_config()['RUNS_DIR']`) to save the results in.
        If it doesn't exist, it is created on the fly.
    """

    # put arguments in right format
    if isinstance(subject, str):
        subjects = [subject]
    else:
        subjects = subject
    if "all" in subjects:
        subjects = [
            "UTS01",
            "UTS02",
            "UTS03",
            "UTS04",
            "UTS05",
            "UTS06",
            "UTS07",
            "UTS08",
        ]

    if isinstance(n_train_stories, int):
        n_train_stories_list = [n_train_stories]
    else:
        n_train_stories_list = n_train_stories

    # pick the right pool of stories, depending on ridge implementation
    if ridge_implementation == "ridge_huth":
        stories = load_config()["STORIES"].copy()
    elif ridge_implementation == "ridgeCV":
        stories = load_config()["STORIES_2"].copy()

    if not isinstance(stories, list):
        raise ValueError(f"Config parameter invalid: STORIES: {stories}")

    if isinstance(feature, str):
        features = [feature]
    else:
        features = feature
    if "all" in features:
        features = ["envelope", "eng1000"]

    # toggle the shuffle swithch
    shuffle_opts = [False]
    if do_shuffle:
        shuffle_opts = [False, True]

    # if run folder name not given, create one
    if not run_folder_name:
        run_folder_name = create_run_folder_name()
    else:
        run_folder_name = run_folder_name

    run_folder = os.path.join(RUNS_DIR, run_folder_name)
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    # log all parameters
    config = {
        "cross_validation": cross_validation,
        "subject_arg": subject,  # command line arguments
        "subjects": subjects,  # resolved predictors
        "feature_arg": feature,
        "features": features,
        "n_train_stories": n_train_stories,
        "stories": stories,
        "test_story": test_story,
        "n_repeats": n_repeats,
        "ndelays": ndelays,
        "interpolation": interpolation,
        "ridge_implementation": ridge_implementation,
        "do_shuffle": shuffle_opts,
        "use_cache": use_cache,
        "keep_train_stories_in_mem": keep_train_stories_in_mem,
        "seed": seed,
        "tr_len": TR_LEN,
        "alphas": list(alphas),
        "nboots": nboots,
        "chunklen": chunklen,
        "nchunks": nchunks,
        "singcutoff": singcutoff,
        "single_alpha": single_alpha,
        "use_corr": use_corr,
        "run_folder_name": run_folder_name,
    }

    log.info(f"Running experiment with the following parameters:\n{json.dumps(config)}")

    # update results file
    params_path = os.path.join(run_folder, "params.json")
    with open(params_path, "w") as f_out:
        json.dump(config, f_out, indent=4)
    log.info(f"Written parameters to {params_path}")

    # aggregate overall max correlations and continously update in json
    results_max_agg = defaultdict(partial(defaultdict, partial(defaultdict, dict)))
    # enables instantiating hierarchy of dicts without manually creating them at each
    # level.

    # get a list of 3-element tuples, with all posible combinations
    combinations = list(product(features, subjects, shuffle_opts))

    results_max_path = os.path.join(run_folder, "results_max.json")

    # run the regression pipeline
    for combination_tuple in combinations:
        current_feature, current_subject, shuffle = combination_tuple

        shuffle_str = "shuffled" if shuffle else "not_shuffled"

        # just for UTS03, nans are not an issue
        if (current_subject == "UTS03") & (ridge_implementation == "ridgeCV"):
            stories = load_config()["STORIES"].copy()
            # config is wrong for UTS03

        for current_n_train_stories in n_train_stories_list:
            output_dir = os.path.join(
                run_folder,
                current_subject,
                current_feature,
                str(current_n_train_stories),
                shuffle_str,
            )
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            if cross_validation == "loocv":
                all_scores = crossval_loocv(
                    feature=current_feature,
                    stories=stories,
                    n_train_stories=current_n_train_stories,
                    subject=current_subject,
                    tr_len=TR_LEN,
                    ndelays=ndelays,
                    interpolation=interpolation,
                    ridge_implementation=ridge_implementation,
                    use_cache=use_cache,
                    shuffle=shuffle,
                    alphas=alphas,
                    nboots=nboots,
                    chunklen=chunklen,
                    nchunks=nchunks,
                    singcutoff=singcutoff,
                    single_alpha=single_alpha,
                    use_corr=use_corr,
                )
            elif cross_validation == "simple":
                all_scores = crossval_simple(
                    feature=current_feature,
                    stories=stories,
                    n_train_stories=current_n_train_stories,
                    test_story=test_story,
                    subject=current_subject,
                    tr_len=TR_LEN,
                    ndelays=ndelays,
                    interpolation=interpolation,
                    ridge_implementation=ridge_implementation,
                    use_cache=use_cache,
                    shuffle=shuffle,
                    n_repeats=n_repeats,
                    seed=seed,
                    keep_train_stories_in_mem=keep_train_stories_in_mem,
                    alphas=alphas,
                    nboots=nboots,
                    chunklen=chunklen,
                    nchunks=nchunks,
                    singcutoff=singcutoff,
                    single_alpha=single_alpha,
                    use_corr=use_corr,
                )
            else:
                raise ValueError(f"Invalid cross validation method: {cross_validation}")

            # aggregate scores
            mean_scores = np.mean(all_scores, axis=0)
            sem_scores = sem(
                [np.mean(repeat_scores, axis=0) for repeat_scores in all_scores], axis=0
            )

            log.info(
                f"Mean correlation (averages across splits) (r): {mean_scores.mean()}"
            )
            log.info(
                f"Max  correlation (averaged across splits) (r): {mean_scores.max()}"
            )

            np.save(os.path.join(output_dir, "scores_mean.npy"), mean_scores)
            np.save(os.path.join(output_dir, "scores_sem.npy"), sem_scores)

            results_max_agg[current_feature][current_subject][current_n_train_stories][
                shuffle_str
            ] = mean_scores.max()

            del mean_scores
            del sem_scores

            # update results file
            with open(results_max_path, "w") as f_out:
                json.dump(results_max_agg, f_out, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_all.py")
    parser.add_argument(
        "--cross_validation", choices=["simple", "loocv"], default="simple"
    )
    parser.add_argument(
        "--subject",
        nargs="+",
        type=str,
        default=["UTS02"],
        choices=[
            "all",
            "UTS01",
            "UTS02",
            "UTS03",
            "UTS04",
            "UTS05",
            "UTS06",
            "UTS07",
            "UTS08",
        ],
    )
    parser.add_argument(
        "--feature",
        nargs="+",
        type=str,
        default=["eng1000"],
        choices=["all", "eng1000", "envelope"],
    )
    parser.add_argument("--n_train_stories", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--test_story", type=str, default="wheretheressmoke")
    parser.add_argument("--n_repeats", default=5, type=int)
    parser.add_argument("--ndelays", type=int, default=5)
    parser.add_argument(
        "--interpolation", type=str, choices=["lanczos", "average"], default="lanczos"
    )
    parser.add_argument(
        "--ridge_implementation",
        type=str,
        choices=["ridgeCV", "ridge_huth"],
        default="ridge_huth",
    )
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--no_keep_train_stories_in_mem", action="store_true")
    parser.add_argument("--nboots", type=int, default=50)
    parser.add_argument("--chunklen", type=int, default=40)
    parser.add_argument("--nchunks", type=int, default=125)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument("--single_alpha", action="store_true")
    parser.add_argument("--use_corr", action="store_true")
    parser.add_argument("--run_folder_name", type=str)
    args = parser.parse_args()

    run_all(
        cross_validation=args.cross_validation,
        subject=args.subject,
        feature=args.feature,
        n_train_stories=args.n_train_stories,
        test_story=args.test_story,
        n_repeats=args.n_repeats,
        ndelays=args.ndelays,
        interpolation=args.interpolation,
        ridge_implementation=args.ridge_implementation,
        do_shuffle=args.do_shuffle,
        use_cache=not args.no_cache,
        keep_train_stories_in_mem=not args.no_keep_train_stories_in_mem,
        nboots=args.nboots,
        chunklen=args.chunklen,
        nchunks=args.nchunks,
        singcutoff=args.singcutoff,
        single_alpha=args.single_alpha,
        use_corr=args.use_corr,
        run_folder_name=args.run_folder_name,
    )
