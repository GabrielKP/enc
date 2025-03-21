import argparse
import os
from collections import defaultdict
from functools import partial
from itertools import product
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import cortex
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cortex import svgoverlay
from rich.console import Console

from encoders.utils import check_make_dirs, get_logger, load_config

console = Console()
log = get_logger(__name__)

# matpltotlib params
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["figure.labelsize"] = 14

SUBJECT_IDS = ["UTS01", "UTS02", "UTS03"]
# this 8-color is also available in seaborn:
# https://seaborn.pydata.org/generated/seaborn.husl_palette.html
# but defined here manually to avoid needing seaborn as dependency
HUSL_PALETTE = [
    "#f77189",
    "#ce9032",
    "#97a431",
    "#32b166",
    "#36ada4",
    "#39a7d0",
    "#a48cf4",
    "#f561dd",
]

# make sure inkscape is installed
INKSCAPE_PATH = load_config().get("INKSCAPE_PATH")
if not Path(INKSCAPE_PATH).exists():
    log.critical(
        "INKSCAPE_PATH not valid. Install inkscape, and place path to"
        " excecutale into the config as INKSCAPE_PATH."
    )
    import sys

    sys.exit(-1)

# add inkscape path to PATH
os.environ["PATH"] = INKSCAPE_PATH + ":" + os.environ["PATH"]
svgoverlay.INKSCAPE_VERSION = str(load_config()["INKSCAPE_VERSION"])


def plot_voxel_performance(
    subject: str,
    scores: np.ndarray,
    ax=None,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 0.5,
    cmap: str = "inferno",
):
    """Plots the given scores for each voxel.

    Parameters
    ----------
    subject : {"UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06", "UTS07", "UTS08"}
        Subject identifier
    scores : np.ndarray
        The scores for each voxel
    save_path : str or None
        Path to save the plot. If `None`, plot will not be saved.
    show_plot : bool, optional
        Whether to show the plot or not.
    title : str, optional
        Title for the plot. If `None`, no title will be shown.
    vmin : float or None, optional
        Minimal value on color map. If `None`, pycortex chooses the value automatically.
    vmax : float or None, optional
        Maximal value on color map. If `None`, pycortex chooses the value automatically.
    cmap : str, optional
        Color map for plot.
    ax : matplotlib.Axis,
    """

    vol_data = cortex.Volume(
        scores, subject, f"{subject}_auto", vmin=vmin, vmax=vmax, cmap=cmap
    )
    # cortex.quickshow(vol_data)

    _ = cortex.quickflat.make_figure(
        braindata=vol_data,
        fig=ax,
        recache=False,
    )

    return None


def load_data(
    run_folder_name: Union[str, Path],
    subject: str,
    feature: str,
    curr_n_train_stories: int,
    shuffle: str,
):
    base_path = Path(
        run_folder_name, subject, feature, str(curr_n_train_stories), shuffle
    )
    fn = Path(base_path, "scores_mean.npy")
    fn2 = Path(base_path, "scores_sem.npy")
    return np.load(fn), np.load(fn2)


def resolve_parameters(
    run_folder_name: str | Path,
    subjects: Optional[Union[str, list[str]]] = None,
    features: Optional[Union[str, list[str]]] = None,
    n_train_stories: Optional[Union[int, list[int]]] = None,
    shuffles: Optional[Union[str, list[str]]] = None,
) -> tuple[list[str], list[str], list[int], list[str]]:
    """Takes any of subject/featres/n_train_stories/shuffles and returns parameters
    that are not specified in run_folder_name."""

    # Handle 'missing' params
    if subjects is None:
        subjects = list()
    if features is None:
        features = list()
    if n_train_stories is None:
        n_train_stories = list()
    if shuffles is None:
        shuffles = list()

    # Handle params not given as list
    if not isinstance(subjects, list):
        subjects = [subjects]
    if not isinstance(features, list):
        features = [features]
    if not isinstance(n_train_stories, list):
        n_train_stories = [n_train_stories]
    if not isinstance(shuffles, list):
        shuffles = [shuffles]

    # 'Discover' params
    if len(subjects) == 0:
        discover_path = Path(run_folder_name)
        discovered_subjects = list(discover_path.glob("UTS[0-9][0-9]"))
        if len(discovered_subjects) == 0:
            raise ValueError(f"No subjects in data folder: {str(discover_path)}")
        subjects = list(map(lambda x: x.name, discovered_subjects))
    if len(features) == 0:
        discover_path = Path(run_folder_name, subjects[0])
        discovered_features = list(discover_path.glob("*"))
        if len(discovered_features) == 0:
            raise ValueError(f"No features in data folder: {str(discover_path)}")
        features = list(map(lambda x: x.name, discovered_features))
    if len(n_train_stories) == 0:
        discover_path = Path(run_folder_name, subjects[0], features[0])
        discovered_n_train_stories = list(discover_path.glob("*"))
        if len(discovered_n_train_stories) is None:
            raise ValueError(f"No n_train_stories in data folder: {str(discover_path)}")
        n_train_stories = list(map(lambda x: int(x.name), discovered_n_train_stories))
    if len(shuffles) == 0:
        discover_path = Path(
            run_folder_name, subjects[0], features[0], str(n_train_stories[0])
        )
        discovered_shuffles = list(discover_path.glob("*"))
        if len(discovered_shuffles) is None:
            raise ValueError(f"No shuffles in data folder: {str(discover_path)}")
        shuffles = list(map(lambda x: x.name, discovered_shuffles))

    return subjects, features, n_train_stories, shuffles


def load_data_wrapper(
    run_folder_name: str | Path,
    subjects: Optional[Union[str, list[str]]] = None,
    features: Optional[Union[str, list[str]]] = None,
    n_train_stories: Optional[Union[int, list[int]]] = None,
    shuffles: Optional[Union[str, list[str]]] = None,
) -> tuple[
    Mapping[str, Mapping[str, Mapping[int, Mapping[str, np.ndarray]]]],
    Mapping[str, Mapping[str, Mapping[int, Mapping[str, np.ndarray]]]],
]:
    """Load data for given configuration. Parameters not given will
    be automatically 'discovered' in the path.
    """

    subjects, features, n_train_stories, shuffles = resolve_parameters(
        run_folder_name, subjects, features, n_train_stories, shuffles
    )

    # get a list of 4-element tuples, with all posible combinations
    combinations = list(product(subjects, features, n_train_stories, shuffles))

    # default_dict & partial enables instantiating hierarchy of dicts without
    # manually creating them at each level.
    rho_means = defaultdict(
        partial(defaultdict, partial(defaultdict, partial(defaultdict, dict)))
    )
    rho_sem = defaultdict(
        partial(defaultdict, partial(defaultdict, partial(defaultdict, dict)))
    )
    found_data = False
    for (
        subject,
        feature,
        curr_n_train_stories,
        shuffle,
    ) in combinations:
        try:
            mean_data, sem_data = load_data(
                run_folder_name=run_folder_name,
                subject=subject,
                feature=feature,
                curr_n_train_stories=curr_n_train_stories,
                shuffle=shuffle,
            )
            rho_means[subject][feature][curr_n_train_stories][shuffle] = mean_data
            rho_sem[subject][feature][curr_n_train_stories][shuffle] = sem_data
            found_data = True
        except FileNotFoundError:
            data_path = str(
                Path(
                    run_folder_name,
                    subject,
                    feature,
                    str(curr_n_train_stories),
                    shuffle,
                )
            )
            log.info(f"Missing data for: {data_path}")

    if not found_data:
        raise ValueError(f"No data found for your parameters in {run_folder_name}")

    return rho_means, rho_sem


def load_data_wrapper_df(
    run_folder_name: str | Path,
    subjects: Optional[Union[str, list[str]]] = None,
    features: Optional[Union[str, list[str]]] = None,
    n_train_stories: Optional[Union[int, list[int]]] = None,
    shuffles: Optional[Union[str, list[str]]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does the same as `load_data_wrapper` but returns two
    dataframes in long format instead.

    """
    subjects, features, n_train_stories, shuffles = resolve_parameters(
        run_folder_name, subjects, features, n_train_stories, shuffles
    )

    # get a list of 4-element tuples, with all posible combinations
    combinations = list(product(subjects, features, n_train_stories, shuffles))

    rho_means_df_ls: list[pd.DataFrame] = list()
    rho_sems_df_ls: list[pd.DataFrame] = list()

    for (
        subject,
        feature,
        curr_n_train_stories,
        shuffle,
    ) in combinations:
        try:
            mean_data, sem_data = load_data(
                run_folder_name=run_folder_name,
                subject=subject,
                feature=feature,
                curr_n_train_stories=curr_n_train_stories,
                shuffle=shuffle,
            )

            rho_means_df_ls.append(
                pd.DataFrame(
                    data=[mean_data],
                    index=pd.MultiIndex.from_tuples(
                        [(subject, feature, curr_n_train_stories, shuffle)]
                    ),
                )
            )
            rho_sems_df_ls.append(
                pd.DataFrame(
                    data=[sem_data],
                    index=pd.MultiIndex.from_tuples(
                        [(subject, feature, curr_n_train_stories, shuffle)]
                    ),
                )
            )
        except FileNotFoundError:
            data_path = str(
                Path(
                    run_folder_name,
                    subject,
                    feature,
                    str(curr_n_train_stories),
                    shuffle,
                )
            )
            log.info(f"Missing data for: {data_path}")

    if len(rho_means_df_ls) == 0:
        raise ValueError(f"No data found for your parameters in {run_folder_name}")

    rho_voxel_means_df = pd.concat(rho_means_df_ls)
    rho_sems_df = pd.concat(rho_sems_df_ls)
    rho_voxel_means_df.index.set_names(
        ("subject", "feature", "curr_n_train_stories", "shuffle"), inplace=True
    )
    rho_sems_df.index.set_names(
        ("subject", "feature", "curr_n_train_stories", "shuffle"), inplace=True
    )
    rho_sems_df.rename(columns={0: "SEM"}, inplace=True)

    return rho_voxel_means_df, rho_sems_df


def make_performance_plots(scores_dict: Dict) -> matplotlib.figure.Figure:
    n_n_train_stories = len(scores_dict)

    fig, ax = plt.subplots(
        1, len(scores_dict), figsize=(len(scores_dict) * 4, 4), layout="constrained"
    )
    if n_n_train_stories == 1:
        ax = [ax]

    for i, items in enumerate(scores_dict.items()):
        n, data = items
        plot_voxel_performance(scores=data, subject="UTS02", vmin=0, vmax=0.5, ax=ax[i])

        ax[i].set_title(
            f"{n} Training story" if int(n) == 1 else f"{n} Training stories"
        )

    if n_n_train_stories > 1:
        cbar = ax[1].images[0].colorbar
    else:
        cbar = ax[0].images[0].colorbar

    cbar.ax.set_xlabel("Test-set correlation", fontsize=12)

    return fig


def make_brain_fig(
    run_folder_name: Union[str, Path],
    subject: str,
    feature: str,
    n_train_stories: list[int],
    shuffle: str = "not_shuffled",
) -> matplotlib.figure.Figure:
    # load data
    rho_means, _ = load_data_wrapper(
        run_folder_name=run_folder_name,
        subjects=subject,
        features=feature,
        n_train_stories=n_train_stories,
        shuffles=shuffle,
    )

    # get it out of the dictionaries
    scores_dict: dict[str, np.ndarray] = dict()
    for curr_n_train_stories in n_train_stories:
        if shuffle in rho_means[subject][feature][curr_n_train_stories]:
            scores_dict[str(curr_n_train_stories)] = rho_means[subject][feature][
                curr_n_train_stories
            ][shuffle]

    fig = make_performance_plots(scores_dict=scores_dict)

    return fig


def make_training_curve_fig(
    run_folder_name: str,
    feature: str,
    subjects: Optional[list[str]],
    n_train_stories: Optional[list[int]],
    shuffle: str,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Plots mean correlation with standard error."""

    assert not isinstance(feature, list), "Can only plot for one feature"
    assert not isinstance(shuffle, list), "Can only plot for one shuffle"

    # load for multiple subjects and n_train_stories
    rho_voxel_means_df, rho_sems_df = load_data_wrapper_df(
        run_folder_name=run_folder_name,
        subjects=subjects,
        features=feature,
        n_train_stories=n_train_stories,
        shuffles=shuffle,
    )

    # compute mean for rho_means_df
    rho_means_df = (
        rho_voxel_means_df.mean(axis=1).to_frame().rename(columns={0: "rho_mean"})
    )
    plot_df = pd.merge(
        rho_means_df, rho_sems_df, left_index=True, right_index=True
    ).reset_index()
    # sort curr_n_train_stories for error bars
    plot_df = plot_df.sort_values(
        ["subject", "feature", "curr_n_train_stories", "shuffle"]
    )

    sns.set_style("ticks")
    sns.set_palette("husl", 8)

    # plot
    ax = sns.lineplot(
        data=plot_df,
        x="curr_n_train_stories",
        y="rho_mean",
        hue="subject",
        ax=ax,
        marker="o",
    )

    # add error bars
    colors = sns.husl_palette(8)
    for idx, subject in enumerate(sorted(plot_df["subject"].unique())):
        subject_df = plot_df[plot_df["subject"] == subject]
        ax.fill_between(
            x=subject_df["curr_n_train_stories"],
            y1=subject_df["rho_mean"] - subject_df["SEM"],
            y2=subject_df["rho_mean"] + subject_df["SEM"],
            color=colors[idx],
            alpha=0.2,
        )

    # styling / text
    ax.set_xlabel("Number of Taining Stories")
    ax.set_ylabel("Mean Correlation (r)")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 0.1)
    # ax.get_legend().set_visible(False)
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)

    return ax


def save_fig_png_pdf(
    fig: matplotlib.figure.Figure,
    save_path: Union[str, Path],
    filename: str,
):
    """Saves figure to pdf and png"""
    fn = str(Path(save_path, f"{filename}.pdf"))
    log.info(f"Saving {fn}")
    fig.savefig(fn, bbox_inches="tight", transparent=True)

    fn_png = fn.replace(".pdf", ".png")
    log.info(f"Saving {fn_png}")
    fig.savefig(fn_png, bbox_inches="tight", dpi=300)
    plt.close()


def plot_all(
    reproduction_folder: Optional[str],
    replication_ridgeCV_folder: Optional[str],
    replication_ridge_huth_folder: Optional[str],
    save_path: Optional[Union[str, Path]],
    main_subject: str = "UTS02",
    n_train_stories_main_subject: list[int] = [1, 11, 25],
):
    """Plots replication, and reproduction plots."""

    plt.ioff()  # turn interactive mode off

    if save_path is None:
        save_path = "plots"

    save_path = Path(save_path)
    check_make_dirs(save_path, isdir=True)

    console.print("\nCorrelation X n_train_stories curve", style="red bold")

    # REPRODUCTION: Training curve
    if reproduction_folder is not None:
        console.print(
            "\n > Reproduction: different-team-same-articacts", style="yellow"
        )
        fig3_reproduction, ax3_reproduction = plt.subplots(figsize=(10, 8))
        make_training_curve_fig(
            run_folder_name=reproduction_folder,
            feature="eng1000",
            subjects=None,
            n_train_stories=None,
            shuffle="not_shuffled",
            ax=ax3_reproduction,
        )
        plt.tight_layout()
        save_fig_png_pdf(
            fig3_reproduction,
            save_path=save_path,
            filename="training_curve_reproduction",
        )

    # REPLICATION ridgeCV: Training curve
    if replication_ridgeCV_folder is not None:
        console.print(
            "\n > Replication ridgeCV: different-team-different-articacts",
            style="yellow",
        )
        fig3_replication_ridgeCV, ax3_replication_ridgeCV = plt.subplots(
            figsize=(10, 8)
        )
        make_training_curve_fig(
            run_folder_name=replication_ridgeCV_folder,
            feature="eng1000",
            subjects=None,
            n_train_stories=None,
            shuffle="not_shuffled",
            ax=ax3_replication_ridgeCV,
        )
        plt.tight_layout()
        save_fig_png_pdf(
            fig3_replication_ridgeCV,
            save_path=save_path,
            filename="training_curve_replication_ridgeCV",
        )

    # REPLICATION ridge_huth: Training curve
    if replication_ridge_huth_folder is not None:
        console.print(
            "\n > Replication ridge_huth: different-team-quasi_same-articacts",
            style="yellow",
        )
        fig3_replication_ridge_huth, ax3_replication_ridge_huth = plt.subplots(
            figsize=(10, 8)
        )
        make_training_curve_fig(
            run_folder_name=replication_ridge_huth_folder,
            feature="eng1000",
            subjects=None,
            n_train_stories=None,
            shuffle="not_shuffled",
            ax=ax3_replication_ridge_huth,
        )
        plt.tight_layout()
        save_fig_png_pdf(
            fig3_replication_ridge_huth,
            save_path=save_path,
            filename="training_curve_replication_ridge_huth",
        )

    console.print("\nBrain fig", style="red bold")
    # REPRODUCTION: Brain fig
    if (
        reproduction_folder is not None
        and Path(reproduction_folder, main_subject, "eng1000").exists()
    ):
        console.print(
            "\n > Reproduction: different-team-same-articacts", style="yellow"
        )
        fig1_reproduction = make_brain_fig(
            run_folder_name=reproduction_folder,
            subject=main_subject,
            feature="eng1000",
            n_train_stories=n_train_stories_main_subject,
            shuffle="not_shuffled",
        )
        fig1_reproduction.suptitle(
            "Reproduction: "
            + "Semantic encoding model performance with increasing training data",
            fontsize=14,
        )
        save_fig_png_pdf(
            fig1_reproduction,
            save_path=save_path,
            filename="reproduction_semantic_performance",
        )

    # REPLICATION ridgeCV: brain fig
    if (
        replication_ridgeCV_folder is not None
        and Path(replication_ridgeCV_folder, main_subject, "eng1000").exists()
    ):
        console.print(
            "\n > Replication ridgeCV: different-team-different-articacts",
            style="yellow",
        )
        fig2_replication_ridgeCV = make_brain_fig(
            run_folder_name=replication_ridgeCV_folder,
            subject=main_subject,
            feature="eng1000",
            n_train_stories=n_train_stories_main_subject,
            shuffle="not_shuffled",
        )
        fig2_replication_ridgeCV.suptitle(
            "Replication: "
            + "Semantic encoding model performance with increasing training data",
            fontsize=14,
        )
        save_fig_png_pdf(
            fig2_replication_ridgeCV,
            save_path=save_path,
            filename="replication_ridgeCV_semantic_performance",
        )

    # REPLICATION ridge_huth: brain fig
    if (
        replication_ridge_huth_folder is not None
        and Path(replication_ridge_huth_folder, main_subject, "eng1000").exists()
    ):
        console.print(
            "\n > Replication ridge_huth: different-team-quasi_same_articacts",
            style="yellow",
        )
        fig2_replication_ridge_huth = make_brain_fig(
            run_folder_name=replication_ridge_huth_folder,
            subject=main_subject,
            feature="eng1000",
            n_train_stories=n_train_stories_main_subject,
            shuffle="not_shuffled",
        )
        fig2_replication_ridge_huth.suptitle(
            "Replication: "
            + "Semantic encoding model performance with increasing training data",
            fontsize=14,
        )
        save_fig_png_pdf(
            fig2_replication_ridge_huth,
            save_path=save_path,
            filename="replication_ridge_huth_semantic_performance",
        )

    console.print("\nBrain fig audio envelope", style="red bold")
    # REPLICATION ridgeCV: brain fig
    if (
        replication_ridgeCV_folder is not None
        and Path(replication_ridgeCV_folder, main_subject, "envelope").exists()
    ):
        console.print(
            "\n > Replication ridgeCV: different-team-different-articacts",
            style="yellow",
        )
        fig2_replication_ridgeCV_envelope = make_brain_fig(
            run_folder_name=replication_ridgeCV_folder,
            subject=main_subject,
            feature="envelope",
            n_train_stories=n_train_stories_main_subject,
            shuffle="not_shuffled",
        )
        fig2_replication_ridgeCV_envelope.suptitle(
            "Extension: "
            + "Sensory encoding model performance with increasing training data",
            fontsize=14,
        )

        save_fig_png_pdf(
            fig2_replication_ridgeCV_envelope,
            save_path=save_path,
            filename="extension_ridgeCV_sensory_performance",
        )

    # REPLICATION ridge_huth: brain fig
    if (
        replication_ridge_huth_folder is not None
        and Path(replication_ridge_huth_folder, main_subject, "envelope").exists()
    ):
        console.print(
            "\n > Replication ridge_huth: different-team-quasi_same-articacts",
            style="yellow",
        )
        fig2_replication_ridge_huth_envelope = make_brain_fig(
            run_folder_name=replication_ridge_huth_folder,
            subject=main_subject,
            feature="envelope",
            n_train_stories=n_train_stories_main_subject,
            shuffle="not_shuffled",
        )
        fig2_replication_ridge_huth_envelope.suptitle(
            "Extension: "
            + "Sensory encoding model performance with increasing training data",
            fontsize=14,
        )

        save_fig_png_pdf(
            fig2_replication_ridge_huth_envelope,
            save_path=save_path,
            filename="extension_ridgeCV_sensory_performance",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "plot.py",
        description="Save replication and reproduction figures as .pdf and .png",
    )
    parser.add_argument(
        "--reproduction",
        type=str,
        default=None,
        help="folder with results for the reproduction experiment to be plotted",
    )
    parser.add_argument(
        "--replication_ridgeCV",
        type=str,
        default=None,
        help="folder with result for the replication experiment to be ploted",
    )
    parser.add_argument(
        "--replication_ridge_huth",
        type=str,
        default=None,
        help="folder with result for the replication experiment to be ploted",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="path to where the figures are saved",
    )

    args = parser.parse_args()

    cfg = load_config()

    plot_all(
        reproduction_folder=args.reproduction,
        replication_ridgeCV_folder=args.replication_ridgeCV,
        replication_ridge_huth_folder=args.replication_ridge_huth,
        save_path=args.save_path,
    )
