import argparse
import json
import os
from collections import defaultdict
from functools import partial
from itertools import product
from typing import Union

import numpy as np

from encoders.main import do_regression
from encoders.utils import (
    check_make_dirs,
    create_run_folder_name,
    get_logger,
    load_config,
)

log = get_logger(__name__)

RUNS_DIR = load_config()["RUNS_DIR"]


def run_all(
    strategy: str = "simple",
    predictor: Union[str, list[str]] = "all",
    n_train_stories: Union[int, list[int]] = [1, 3, 5],
    n_repeats: int = 5,
    subject: Union[str, list[str]] = "UTS02",
    n_delays: int = 5,
    interpolation: str = "lanczos",
    ridge_implementation: str = "ridge_huth",
    do_shuffle: bool = False,
    use_cache: bool = True,
    keep_train_stories_in_mem: bool = True,
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
    strategy : {"loocv", "simple"}, default="simple"
        `loocv` uses leave-one-out cross-validation for n_stories. The stories are
         determined by the order of the `stories` parameter or its default value in
         `config.yaml`.
        `simple` computes the regression for a train/test split containing n_stories
         within each repeat.
        Stories are sampled randomly for each repeat.
    predictor : {"all", "envelope", "embeddings"}, default="all"
        Which predictor to run.
        `all` will run separate encoding models for both predictors (default).
        `envelope` will run the encoding model with the audio envelope as predictor.
        `embeddings` will run the encoding model with the word embeddings of the stories
         as predictor.
    n_train_stories : int or list of int
        Number o of training stories for the encoding model. If a list is given, the
         encoding model will be fitted with each number separately.
    subject : str or list of str, default="UTS02"
        Subject identifier.
        Can be one or a list of : {`"all"`, `"UTS01"`, `"UTS02"`, `"UTS03"`, `"UTS04"`,
         `"UTS05"`, `"UTS06"`, `"UTS07"`, `"UTS08"`}
    n_delays : int, default=5
        By how many TR's features are delayed to model the HRF. For `n_delays=5`, the
         features of the predictor are shifted by one TR and concatinated to themselves
         for five times.
    interpolation : {"lanczos", "average"}, default="lanczos"
        Whether to use lanczos interpolation or just average the words within a TR.
        Only applies if `predictor=embeddings`.
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
        time is spent loading data. Only works if `strategy='simple'`.
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

    if isinstance(predictor, str):
        predictors = [predictor]
    else:
        predictors = predictor
    if "all" in predictors:
        predictors = ["envelope", "embeddings"]

    # toggle the shuffle swithch
    shuffle_opts = [False]
    if do_shuffle:
        shuffle_opts = [False, True]

    # handle data folder
    folder_name = create_run_folder_name()
    base_dir = os.path.join(RUNS_DIR, folder_name)
    check_make_dirs(base_dir, isdir=True)

    # log all parameters
    config = {
        "strategy": strategy,
        "predictor_arg": predictor,
        "predictors": predictors,
        "n_train_stories": n_train_stories,
        "n_repeats": n_repeats,
        "subject_arg": subject,  # command line arguments
        "subjects": subjects,  # resolved predictors
        "n_delays": n_delays,
        "interpolation": interpolation,
        "ridge_implementation": ridge_implementation,
        "do_shuffle": shuffle_opts,
        "use_cache": use_cache,
        "keep_train_stories_in_mem": keep_train_stories_in_mem,
    }

    log.info(f"Running experiment with the following parameters:\n{json.dumps(config)}")

    # update results file
    params_path = os.path.join(base_dir, "params.json")
    with open(params_path, "w") as f_out:
        json.dump(config, f_out, indent=4)
    log.info(f"Written parameters to {params_path}")

    # aggregate overall max correlations and continously update in json
    results_max_agg = defaultdict(partial(defaultdict, partial(defaultdict, dict)))
    # enables instantiating hierarchy of dicts without manually creating them at each
    # level.

    # get a list of 3-element tuples, with all posible combinations
    combinations = list(product(predictors, subjects, shuffle_opts))

    results_max_path = os.path.join(base_dir, "results_max.json")

    for combination_tuple in combinations:
        current_predictor, current_subject, shuffle = combination_tuple

        shuffle_str = "shuffled" if shuffle else "not_shuffled"

        for current_n_train_stories in n_train_stories_list:
            output_dir = os.path.join(
                base_dir,
                current_predictor,
                current_subject,
                str(current_n_train_stories),
                shuffle_str,
            )
            check_make_dirs(output_dir, verbose=False, isdir=True)
            mean_scores, _, _, _ = do_regression(
                strategy=strategy,
                predictor=current_predictor,
                n_train_stories=current_n_train_stories,
                n_repeats=n_repeats,
                subject=current_subject,
                n_delays=n_delays,
                interpolation=interpolation,
                ridge_implementation=ridge_implementation,
                use_cache=use_cache,
                shuffle=shuffle,
                show_results=False,
                keep_train_stories_in_mem=keep_train_stories_in_mem,
            )
            np.save(os.path.join(output_dir, "scores_mean.npy"), mean_scores)

            results_max_agg[current_predictor][current_subject][
                current_n_train_stories
            ][shuffle_str] = mean_scores.max()

            # update results file
            with open(results_max_path, "w") as f_out:
                json.dump(results_max_agg, f_out, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "run_all.py",
        description="",
    )
    parser.add_argument(
        "--strategy",
        choices=["simple", "loocv"],
        default="loocv",
        help=(
            "Whether to sample train/test n_repeat times (simple) or do"
            " leave one out cross validation (loocv)"
        ),
    )
    parser.add_argument(
        "--predictor",
        nargs="+",
        type=str,
        default=["all"],
        choices=["all", "embeddings", "envelope"],
        help=(
            "Predictor for the encoding model. Can be 'all' or a combination of"
            " predictors."
        ),
    )
    parser.add_argument(
        "--n_train_stories",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help=(
            "Only used if `strategy='simple'`."
            " Amount of training stories, can be one or multiple amounts."
        ),
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
        nargs="+",
        type=str,
        default=["UTS02"],
        help="List of subject identifier. Can be 'all' for all subjects.",
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
        "--n_delays",
        type=int,
        default=5,
        help="How many delays are used to model the HRF.",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        choices=["lanczos", "average"],
        default="lanczos",
        help="Interpolation method used for embeddings predictor.",
    )
    parser.add_argument(
        "--ridge_implementation",
        type=str,
        choices=["ridgeCV", "ridge_huth"],
        default="ridge_huth",
        help="Which ridge regression implementation to use.",
    )
    parser.add_argument(
        "--do_shuffle",
        action="store_true",
        help="Whether or not to fit models with randomly shuffled predictors",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Whether the cache is used for `envelope` features.",
    )
    parser.add_argument(
        "--no_keep_train_stories_in_mem",
        action="store_true",
        help="Whether stories are kept in memory after first loading.",
    )
    args = parser.parse_args()
    run_all(
        strategy=args.strategy,
        predictor=args.predictor,
        n_train_stories=args.n_train_stories,
        n_repeats=args.n_repeats,
        subject=args.subject,
        n_delays=args.n_delays,
        interpolation=args.interpolation,
        ridge_implementation=args.ridge_implementation,
        do_shuffle=args.do_shuffle,
        use_cache=not args.no_cache,
        keep_train_stories_in_mem=not args.no_keep_train_stories_in_mem,
    )
