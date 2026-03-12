import argparse
import json
import logging
import os
from os.path import dirname, join
from pathlib import Path
from typing import Optional, Union, cast

import h5py
import numpy as np
from scipy.stats import sem

from encoders.utils import check_make_dirs, create_run_folder_name, load_config
from lebel_encoding.encoding_utils import apply_zscore_and_hrf, get_response
from lebel_encoding.feature_spaces import _FEATURE_CONFIG, get_feature_space
from lebel_encoding.ridge_utils.ridge import bootstrap_ridge

# mofidied from encoding.py to match same parameters as encoders/run_all.py
# https://github.com/HuthLab/deep-fMRI-dataset/blob/master/encoding/encoding.py

cfg = load_config()


RUNS_DIR = cfg["RUNS_DIR"]
STORIES = cfg["STORIES"]


def run_all_replication(
    subject: str = "UTS02",
    feature: str = "eng1000",
    n_train_stories: Union[int, list[int]] = [1, 3, 5],
    test_story: str = "wheretheressmoke",
    n_repeats: int = 15,
    trim: int = 5,
    ndelays: int = 5,
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
    subject : str, default="UTS02"
        Subject identifier.
        Can be one of: {`"all"`, `"UTS01"`, `"UTS02"`, `"UTS03"`, `"UTS04"`,
         `"UTS05"`, `"UTS06"`, `"UTS07"`, `"UTS08"`}
    feature : {"eng1000", "articulation", "phonemerate", "wordrate"}, default="all"
        Which feature to run.
        `eng1000` will run the encoding model with the word embeddings of the stories
         as predictor.
        The other features have not been tested in this codebase (but may work).
    n_train_stories : int or list of int
        Number o of training stories for the encoding model. If a list is given, the
         encoding model will be fitted with each number separately.
    test_story : str, default="wheretheressmoke"
        The story to use as the test set.
    n_repeats : int, default=5
        Determines how often regression is repeated on a different train/test set.
    trim : int, default=5
        Trimming of the features to match preprocessed fmri data.
    ndelays : int, default=5
        By how many TR's features are delayed to model the HRF. For `ndelays=5`, the
         features of the predictor are shifted by one TR and concatenated to themselves
         for five times.
    nboots : int
        The number of bootstrap samples to run. 15 to 30 works well.
    chunklen : int
        On each sample, the training data is broken into chunks of this length. This
        should be a few times longer than your delay/STRF. e.g. for a STRF with 3
        delays, I use chunks of length 10.
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
    run_folder_name: str, optional
        The name of the folder in the runs directory (as specified in
        `encoders.utils.load_config()['RUNS_DIR']`) to save the results in.
        If it doesn't exist, it is created on the fly.
    """

    fs = " ".join(_FEATURE_CONFIG.keys())
    assert feature in _FEATURE_CONFIG.keys(), "Available feature spaces:" + fs

    config = {
        "subject": subject,
        "feature": feature,
        "n_train_stories": n_train_stories,
        "stories": STORIES,
        "test_story": test_story,
        "n_repeats": n_repeats,
        "trim": trim,
        "ndelays": ndelays,
        "nboots": nboots,
        "chunklen": chunklen,
        "nchunks": nchunks,
        "singcutoff": singcutoff,
        "single_alpha": single_alpha,
        "use_corr": use_corr,
        "run_folder_name": run_folder_name,
    }

    run_folder = os.path.join(RUNS_DIR, run_folder_name)
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    # update results file
    params_path = os.path.join(run_folder, "params.json")
    with open(params_path, "w") as f_out:
        json.dump(config, f_out, indent=4)

    test_stories = [test_story]
    train_stories = list(set(STORIES).difference(test_stories))
    allstories = list(set(train_stories) | set(test_stories))

    if isinstance(n_train_stories, int):
        n_train_stories_list = [n_train_stories]
    else:
        n_train_stories_list = n_train_stories

    # if run folder name not given, create one
    if not run_folder_name:
        run_folder_name = create_run_folder_name()
    else:
        run_folder_name = run_folder_name

    for current_n_train_stories in n_train_stories_list:
        # e.g. /UTS01/embeddings/5/not_shuffled/scores_mean.py
        output_dir = os.path.join(
            run_folder,
            subject,
            feature,
            str(current_n_train_stories),
            "not_shuffled",
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print("Saving encoding model & results to:", output_dir)

        downsampled_feat = get_feature_space(feature, allstories)
        print("Stimulus & Response parameters:")
        print("trim: %d, ndelays: %d" % (trim, ndelays))

        # result arrays
        scores_list = list()
        weights_list = list()
        best_alphas_list = list()

        rng = np.random.default_rng()
        for repeat in range(n_repeats):
            print("Repeat %d" % repeat)
            print(train_stories)
            # choose stories to sample for this repeat
            curr_train_stories = cast(
                list[str],
                rng.choice(
                    train_stories, size=current_n_train_stories, replace=False
                ).tolist(),
            )
            curr_test_stories = test_stories

            # Delayed stimulus
            delRstim = apply_zscore_and_hrf(
                curr_train_stories, downsampled_feat, trim, ndelays
            )
            print("delRstim: ", delRstim.shape)
            delPstim = apply_zscore_and_hrf(
                curr_test_stories, downsampled_feat, trim, ndelays
            )
            print("delPstim: ", delPstim.shape)

            # Response
            zRresp = get_response(curr_train_stories, subject)
            print("zRresp: ", zRresp.shape)
            zPresp = get_response(curr_test_stories, subject)
            print("zPresp: ", zPresp.shape)

            # Ridge
            alphas = np.logspace(1, 3, 10)

            print("Ridge parameters:")
            print(
                "nboots: %d, chunklen: %d, nchunks: %d, single_alpha: %s, use_corr: %s"
                % (
                    nboots,
                    chunklen,
                    nchunks,
                    single_alpha,
                    use_corr,
                )
            )

            wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
                delRstim,
                zRresp,
                delPstim,
                zPresp,
                alphas,
                nboots,
                chunklen,
                nchunks,
                singcutoff=singcutoff,
                single_alpha=single_alpha,
                use_corr=use_corr,
            )

            scores_list.append(corrs)
            weights_list.append(wt)
            best_alphas_list.append(valphas)

        mean_scores = np.mean(scores_list, axis=0)
        sem_scores = sem(
            [np.mean(repeat_scores, axis=0) for repeat_scores in scores_list], axis=0
        )
        np.save(os.path.join(output_dir, "scores_mean.npy"), mean_scores)
        np.save(os.path.join(output_dir, "scores_sem.npy"), sem_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="UTS02")
    parser.add_argument("--feature", type=str, default="eng1000")
    parser.add_argument("--n_train_stories", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--test_story", type=str, default="wheretheressmoke")
    parser.add_argument("--n_repeats", type=int, default=15)
    parser.add_argument("--trim", type=int, default=5)
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--nboots", type=int, default=50)
    parser.add_argument("--chunklen", type=int, default=40)
    parser.add_argument("--nchunks", type=int, default=125)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument("--single_alpha", action="store_true")
    parser.add_argument("--use_corr", action="store_true")
    parser.add_argument("--run_folder_name", type=str)
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    run_all_replication(
        subject=args.subject,
        feature=args.feature,
        n_train_stories=args.n_train_stories,
        test_story=args.test_story,
        n_repeats=args.n_repeats,
        trim=args.trim,
        ndelays=args.ndelays,
        nboots=args.nboots,
        chunklen=args.chunklen,
        nchunks=args.nchunks,
        singcutoff=args.singcutoff,
        single_alpha=args.single_alpha,
        use_corr=args.use_corr,
        run_folder_name=args.run_folder_name,
    )
