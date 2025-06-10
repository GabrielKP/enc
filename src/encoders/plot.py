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
from tqdm import tqdm

from encoders.utils import check_make_dirs, get_logger, load_config

console = Console()
log = get_logger(__name__)


SUBJECT_IDS = ["UTS01", "UTS02", "UTS03"]
PALETTE = sns.husl_palette(8)

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


# seaborn settings
sns.set_theme(
    style="ticks",
    palette=PALETTE,
    font="Verdana",
)
# matpltotlib settings
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["figure.labelsize"] = 14
mpl.rcParams["font.family"] = "DeJavu Serif"
mpl.rcParams["font.sans-serif"] = "Verdana"


def plot_voxel_performance(
    subject: str,
    scores: np.ndarray,
    ax=None,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 0.5,
    cmap: str = "inferno",
    **kwargs,
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

    # https://gallantlab.org/pycortex/generated/cortex.quickflat.make_figure.html
    _ = cortex.quickflat.make_figure(
        braindata=vol_data,
        fig=ax,
        recache=False,
        **kwargs,
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
    ) in tqdm(combinations, desc="(loading data)"):
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
    ) in tqdm(combinations, desc="(loading data)"):
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


def make_performance_plots(
    scores_dict: Dict,
    subject: str,
    ax_titles: bool,
    **kwargs,
) -> matplotlib.figure.Figure:
    n_n_train_stories = len(scores_dict)

    fig, ax = plt.subplots(
        1, len(scores_dict), figsize=(len(scores_dict) * 4, 4), layout="constrained"
    )
    if n_n_train_stories == 1:
        ax = [ax]

    for i, items in enumerate(scores_dict.items()):
        n, data = items
        plot_voxel_performance(
            scores=data, subject=subject, vmin=0, vmax=0.5, ax=ax[i], **kwargs
        )

        if ax_titles:
            ax[i].set_title(
                f"{n} Training story" if int(n) == 1 else f"{n} Training stories"
            )

    if kwargs.get("with_colorbar"):
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
    ax_titles: bool = True,
    **kwargs,
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

    fig = make_performance_plots(
        scores_dict=scores_dict,
        subject=subject,
        ax_titles=ax_titles,
        **kwargs,
    )

    return fig


def make_colorbar(
    ax: matplotlib.axes.Axes,
    vmin: float = 0,
    vmax: float = 0.5,
    cmap: str = "inferno",
) -> matplotlib.axes.Axes:
    # Create a horizontal colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # type: ignore
    cbar = mpl.colorbar.ColorbarBase(  # type: ignore
        ax,
        cmap=cmap,
        norm=norm,
        orientation="horizontal",
    )

    # Set labels and ticks
    cbar.outline.set_visible(False)
    cbar.set_label("Correlation (r)", labelpad=-10, fontsize=18)
    cbar.set_ticks([0, 0.5], labels=["0", "0.5"])
    cbar.ax.tick_params(size=0, axis="x", pad=10, labelsize=18)

    return ax


def make_training_curve_fig(
    run_folder_name: str,
    feature: str,
    subjects: Optional[list[str]],
    n_train_stories: Optional[list[int]],
    shuffle: Union[str, list[str]],
    ax: Optional[matplotlib.axes.Axes] = None,
    plot_config: Optional[dict] = None,
) -> matplotlib.axes.Axes:
    """Plots mean correlation with standard error."""

    assert not isinstance(feature, list), "Can only plot for one feature"

    if plot_config is None:
        plot_config = dict()

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

    # plot
    ax = sns.lineplot(
        data=plot_df,
        x="curr_n_train_stories",
        y="rho_mean",
        hue="subject",
        style=plot_config.get("style", None),
        ax=ax,
        marker=plot_config.get("marker", "o"),
        markers=plot_config.get("markers", None),
        markersize=9,
    )

    # add error bars
    for idx, subject in enumerate(sorted(plot_df["subject"].unique())):
        for shuffle in plot_df["shuffle"].unique():
            subject_df = plot_df[
                (plot_df["subject"] == subject) & (plot_df["shuffle"] == shuffle)
            ]
            ax.fill_between(
                x=subject_df["curr_n_train_stories"],
                y1=subject_df["rho_mean"] - subject_df["SEM"],
                y2=subject_df["rho_mean"] + subject_df["SEM"],
                color=PALETTE[idx],
                alpha=0.2,
            )

    # styling / text
    ax.set_xlabel(plot_config.get("xlabel", "Number of Training Stories"), fontsize=18)
    ax.set_ylabel("Mean Correlation (r)", fontsize=18)
    ax.set_xlim(plot_config.get("xlim", (0, 25.35)))
    ax.set_ylim(plot_config.get("ylim", (0.01, 0.09)))
    ax.set_yticks(
        plot_config.get("yticks", [0.02, 0.04, 0.06, 0.08]),
        labels=plot_config.get("ylabels", ["0.02", "0.04", "0.06", "0.08"]),
    )
    ax.set
    ax.grid(visible=True, axis="y", color="#CFCFCF")
    ax.set_axisbelow(True)
    ax.get_legend().set_visible(False)
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

    fn_svg = fn.replace(".pdf", ".svg")
    log.info(f"Saving {fn_svg}")
    fig.savefig(fn_svg, bbox_inches="tight")
    plt.close()


def plot_figure1(
    reproduction_dir: str,
    replication_ridgeCV_dir: str,
    save_path: Optional[Union[str, Path]],
):
    """Plot figure 1 plots"""

    subject = "UTS02"
    figsize = (6, 4)
    # setting the theme twice with seaborn makes it create
    # slightly different plots

    if save_path is None:
        save_path = Path("plots", "figure1")
    check_make_dirs(save_path, isdir=True)

    console.print("\nFigure 1 - 'Main': Reproduction & replication", style="red bold")
    # REPRODUCTION: Training curve
    if Path(reproduction_dir).exists():
        console.print(
            "\n > Training curve - Reproduction: different-team-same-articacts",
            style="yellow",
        )
        fig_reproduction, ax_reproduction = plt.subplots(figsize=figsize)
        make_training_curve_fig(
            run_folder_name=reproduction_dir,
            feature="eng1000",
            subjects=None,
            n_train_stories=None,
            shuffle="not_shuffled",
            ax=ax_reproduction,
        )
        plt.tight_layout()
        save_fig_png_pdf(
            fig_reproduction,
            save_path=save_path,
            filename="training_curve_reproduction",
        )
    else:
        log.warning(f"Cannot find reproduction dir: '{Path(reproduction_dir)}'")

    # REPLICATION ridgeCV: Training curve
    if Path(replication_ridgeCV_dir).exists():
        console.print(
            "\n > Training curve - Replication ridgeCV:"
            " different-team-different-articacts",
            style="yellow",
        )
        fig_replication_ridgeCV, ax_replication_ridgeCV = plt.subplots(figsize=figsize)
        make_training_curve_fig(
            run_folder_name=replication_ridgeCV_dir,
            feature="eng1000",
            subjects=None,
            n_train_stories=None,
            shuffle="not_shuffled",
            ax=ax_replication_ridgeCV,
        )
        plt.tight_layout()
        save_fig_png_pdf(
            fig_replication_ridgeCV,
            save_path=save_path,
            filename="training_curve_replication_ridgeCV",
        )
    else:
        log.warning(f"Cannot find replication dir: '{Path(replication_ridgeCV_dir)}'")

    # REPRODUCTION: Brain fig
    if Path(reproduction_dir).exists():
        console.print(
            "\n > Brain fig - Reproduction: different-team-same-articacts",
            style="yellow",
        )
        fig_brain_reproduction = make_brain_fig(
            run_folder_name=reproduction_dir,
            subject=subject,
            feature="eng1000",
            n_train_stories=[25],
            shuffle="not_shuffled",
            ax_titles=False,
            with_colorbar=False,
            with_labels=False,
        )
        save_fig_png_pdf(
            fig_brain_reproduction,
            save_path=save_path,
            filename="reproduction_semantic_performance",
        )

    # REPLICATION ridgeCV: brain fig
    if Path(replication_ridgeCV_dir).exists():
        console.print(
            "\n > Brain fig - Replication ridgeCV: different-team-different-articacts",
            style="yellow",
        )
        fig_brain_replication_ridgeCV = make_brain_fig(
            run_folder_name=replication_ridgeCV_dir,
            subject=subject,
            feature="eng1000",
            n_train_stories=[25],
            shuffle="not_shuffled",
            ax_titles=False,
            with_colorbar=False,
            with_labels=False,
        )
        save_fig_png_pdf(
            fig_brain_replication_ridgeCV,
            save_path=save_path,
            filename="replication_ridgeCV_semantic_performance",
        )

    console.print("\n > Colorbar", style="yellow")
    fig_cbar, ax = plt.subplots(figsize=(6, 0.45))
    make_colorbar(ax)
    save_fig_png_pdf(
        fig=fig_cbar,
        save_path=save_path,
        filename="colorbar",
    )


def plot_figure2(
    replication_ridgeCV_dir: str,
    replication_ridge_huth_dir: str,
    save_path: Optional[Union[str, Path]],
):
    """Plot figure 2 plots"""

    figsize = (5, 4)

    if save_path is None:
        save_path = Path("plots", "figure2")
    check_make_dirs(save_path, isdir=True)
    console.print("\nFigure 2 - 'Replication': Patching experiment", style="red bold")

    plot_config = dict(xlabel="# Training Stories", xlim=(0, 25.9))

    # ridgeCV: Training curve
    console.print("\n > Training curve - ridgeCV", style="yellow")
    fig3_ridgeCV, ax3_ridgeCV = plt.subplots(figsize=figsize)
    make_training_curve_fig(
        run_folder_name=replication_ridgeCV_dir,
        feature="eng1000",
        subjects=None,
        n_train_stories=None,
        shuffle="not_shuffled",
        ax=ax3_ridgeCV,
        plot_config=plot_config,
    )
    plt.tight_layout()
    save_fig_png_pdf(
        fig3_ridgeCV,
        save_path=save_path,
        filename="training_curve_ridgeCV",
    )

    # ridge_huth: Training curve
    console.print("\n > Training curve - ridge_huth", style="yellow")
    fig3_ridge_huth, ax3_ridge_huth = plt.subplots(figsize=figsize)
    make_training_curve_fig(
        run_folder_name=replication_ridge_huth_dir,
        feature="eng1000",
        subjects=None,
        n_train_stories=None,
        shuffle="not_shuffled",
        ax=ax3_ridge_huth,
        plot_config=plot_config,
    )
    plt.tight_layout()
    save_fig_png_pdf(
        fig3_ridge_huth,
        save_path=save_path,
        filename="training_curve_ridge_huth",
    )


def plot_figure3(
    extension_ridgeCV_dir: str,
    save_path: Optional[Union[str, Path]] = None,
):
    console.print("\nFigure 3 - 'Extension': Audio envelope", style="red bold")
    subject = "UTS02"
    figsize = (5, 4)

    if save_path is None:
        save_path = Path("plots", "figure3")
    check_make_dirs(save_path, isdir=True)

    console.print("\n > Training curve - ridge_huth:", style="yellow")
    fig4_extension_curve, ax4_extension_curve = plt.subplots(figsize=figsize)
    make_training_curve_fig(
        run_folder_name=extension_ridgeCV_dir,
        feature="envelope",
        subjects=None,
        n_train_stories=None,
        shuffle=["not_shuffled", "shuffled"],
        ax=ax4_extension_curve,
        plot_config=dict(
            style="shuffle",
            marker=None,
            markers=dict(not_shuffled="o", shuffled="X"),
            ylim=(-0.006, 0.027),
            yticks=[-0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025],
            ylabels=["-.005", ".000", ".005", ".010", ".015", ".020", ".025"],
            xlabel="# Training Stories",
            xlim=(0, 25.9),
        ),
    )
    plt.tight_layout()
    save_fig_png_pdf(
        fig4_extension_curve,
        save_path=save_path,
        filename="training_curve_extension_ridgeCV",
    )

    console.print("\n > Brain fig - ridge_huth", style="yellow")
    fig4_extension_brain = make_brain_fig(
        run_folder_name=extension_ridgeCV_dir,
        subject=subject,
        feature="envelope",
        n_train_stories=[25],
        shuffle="not_shuffled",
        ax_titles=False,
        with_colorbar=False,
        with_labels=False,
    )
    save_fig_png_pdf(
        fig4_extension_brain,
        save_path=save_path,
        filename="semantic_performance_extension_ridgeCV",
    )

    fig_cbar, ax = plt.subplots(figsize=(6, 0.45))
    make_colorbar(ax)

    console.print("\n > Colorbar", style="yellow")
    save_fig_png_pdf(
        fig=fig_cbar,
        save_path=save_path,
        filename="colorbar",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "plot.py",
        description="Save replication and reproduction figures as .pdf and .png",
    )
    parser.add_argument(
        "--reproduction",
        type=str,
        default="runs/reproduction",
        help="folder with correlation results for the reproduction experiment",
    )
    parser.add_argument(
        "--replication_ridgeCV",
        type=str,
        default="runs/replication_ridgeCV",
        help="folder with correlation result for the replication experiment ",
    )
    parser.add_argument(
        "--replication_ridge_huth",
        type=str,
        default="runs/replication_ridge_huth",
        help="folder with correlation result for the replication experiment",
    )
    parser.add_argument(
        "--extension_ridgeCV",
        type=str,
        default="runs/extension_ridgeCV",
        help="folder with correlation result for the extension experiment",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="path to where the figures are saved",
    )
    parser.add_argument(
        "--figure",
        choices=["all", "figure1", "figure2", "figure3"],
        default="all",
        help="Which figures to plot. Default 'all'",
    )

    args = parser.parse_args()

    if args.figure in ["figure1", "all"]:
        plot_figure1(
            reproduction_dir=args.reproduction,
            replication_ridgeCV_dir=args.replication_ridgeCV,
            save_path=args.save_path,
        )
    if args.figure in ["figure2", "all"]:
        plot_figure2(
            replication_ridgeCV_dir=args.replication_ridgeCV,
            replication_ridge_huth_dir=args.replication_ridge_huth,
            save_path=args.save_path,
        )
    if args.figure in ["figure3", "all"]:
        plot_figure3(
            extension_ridgeCV_dir=args.extension_ridgeCV,
            save_path=args.save_path,
        )
