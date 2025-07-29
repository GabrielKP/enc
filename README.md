<h1 align="center">Encoding Models</h1>

<p align="center">A project reproducing & replicating endocing models published by <a href="https://github.com/HuthLab/deep-fMRI-dataset"><i>LeBel et al. 2023</i></a>.</p>
<p align="center">We documented our results in this <a href="https://kristijanarmeni.github.io/encoders_report/"><i>Report</i></a>.</p>

<p align="center">
<a href="https://www.python.org/"><img alt="code" src="https://img.shields.io/badge/code-Python-blue?logo=Python"></a>
<a href="https://gabrielkp.com/enc/"><img alt="documentation" src="https://img.shields.io/badge/docs-MkDocs-708FCC.svg?style=flat"></a>
<a href="https://kristijanarmeni.github.io/encoders_report/"><img alt="documentation" src="https://img.shields.io/badge/Report-MystMD-white?logo=Markdown"></a>
<a href="https://scikit-learn.org/stable/"><img alt="ML framework" src="https://img.shields.io/badge/ML-Scikit%20Learn-orange?logo=Scikit-learn"></a>
<a href="https://docs.astral.sh/ruff/"><img alt="Code style: Ruff" src="https://img.shields.io/badge/code%20style-Ruff-green?logo=Ruff"></a>
<a href="https://python-poetry.org/"><img alt="packaging framwork: Poetry" src="https://img.shields.io/badge/packaging-Poetry-lightblue?logo=Poetry"></a>
<a href="https://pre-commit.com/"><img alt="tool: pre-commit" src="https://img.shields.io/badge/tool-Pre%20Commit-yellow?logo=Pre-Commit"></a>
</p>

---

## Reproducing the published figures

This guide shows how to reproduce the figures from the precomputed model performance scores from all experiments (model-brain correlations) using the provided code.
It requires the installation of the relevant dependencies and the correlation results of various encoding model runs.

#### 1. Clone repository, navigate into created repository directory

```sh
git clone git@github.com:GabrielKP/enc.git
cd enc
```

#### 2. Setup a virtual environment and install dependencies

Next, create a virtual environment with Python 3.12 using your preferred manager.

For example, using conda:

```sh
# conda environment
conda create -n enc python=3.12
conda activate enc
```

Or using [uv](https://docs.astral.sh/uv/) (in project root directory):

```sh
uv venv --python 3.12 --seed # creates .venv environment folder and installs python and pip
source .venv/bin/activate # activate the environment
```

Now that you have python and pin ready, install the project code and its dependencies:
```sh
# install package
pip install .

# install git-annex (required to download data from Lebel et al. 2011)

```

#### 3. Install [git-annex and datalad](https://handbook.datalad.org/en/latest/intro/installation.html) (required to download data from Lebel et al. 2023)

#### 4. Download the subject-specific cortical surface data

You can use our wrapper script that downloads the data from [the original fMRI data repository](https://github.com/OpenNeuroDatasets/ds003020.git) with Datalad:

```sh
# Only download data required for plotting results
python src/encoders/download_data.py --figures [--data_dir DATA_DIR]
```

Without `--data_dir DATA_DIR` the data will be downloaded into the folder `ds003020` of the project directory.
To download the data into a custom dir, specify `--data_dir DATA_DIR` (it is recommended to call the last folder `ds003020` as that is the default dataset name).

#### 5. Setup/check `config.yaml`.

The `config.yaml` should be created automatically when you run the download script in step 4., if you can copy the [example config file](https://github.com/GabrielKP/enc/blob/main/config.example.yaml). Make sure that the `DATA_DIR` points to the folder where you downloaded the data to in step 4.:

```yaml
CACHE_DIR: .cache
DATA_DIR: ds003020 # <-- should point to the download location
RUNS_DIR: runs
INKSCAPE_PATH: /path/to/inkscape
INKSCAPE_VERSION: X.Y.Z
TR_LEN: 2.0
```

> [!NOTE]
> If you get the error `ImportError: cannot import name 'getargspec' from 'inspect'` then try to update your datalad version `python -m pip install datalad --upgrade`

#### 6. Download the pre-computed experiment results (model performance scores)

Download [`runs.zip`](https://osf.io/download/g9cy3) and unzip it such that you have a `runs` directory which contains the results of individual experiments in separate subfolders:

```
runs/extension_ridgeCV
runs/replication_ridge_huth
runs/replication_ridgeCV
runs/reproduction
```

> [!NOTE]
> The `runs` folder should be placed in the project root (i.e. at the same level as `data`, `src` etc.)

#### 7. Install [inkscape](https://inkscape.org/) (required for plotting):

Open the `config.yaml` and set the following values accordingly.

```yaml
INKSCAPE_PATH: path/to/inkscape/binary
INKSCAPE_VERSION: X.Y.Z
```

For mac, you usually can [find inkscape as described here](https://stackoverflow.com/a/22085247).

#### 8. Configure pycortex (required for plotting):

**Using the provided script**

You can use the script in the repository to configure the pycortex:

```sh
python src/encoders/update_pycortex_config.py
```

**Configure manually**

Find the location of your pycortext config with the python terminal.
Type `python` in the command line with the virtual environment activated.
Then execute following commands:

```py
import cortex
cortex.options.usercfg
```

This should give you a path to the config file, copy it and exit the terminal.

```sh
# open the file with an editor of choice (e.g. vim)
vim path/to/options.cfg
```

Modify the entry at `filestore` to `DATA_DIR/derivative/pycortex-db`.
Whereas `DATA_DIR` is the directory of the Lebel et al. data repository.
E.g. if you did not specify a custom datadir, `DATA_DIR` then it is `/path/to/this/repository/ds003020`.

#### 9. Run the script that plots the data:

You can now run the plotting script. From the project root folder call:

```sh
# will create plots for all figures
python src/encoders/plot.py
```

The script creates a `plots` folder in the project root folder with three subfolders, one for each published figure:

```sh
./plots
├── figure1
│   ├── colorbar.pdf
│   ├── colorbar.png
│   ├── colorbar.svg
│   ├── replication_ridgeCV_semantic_performance.pdf
│   ├── replication_ridgeCV_semantic_performance.png
│   ├── replication_ridgeCV_semantic_performance.svg
│   ├── reproduction_semantic_performance.pdf
│   ├── reproduction_semantic_performance.png
│   ├── reproduction_semantic_performance.svg
│   ├── training_curve_replication_ridgeCV.pdf
│   ├── training_curve_replication_ridgeCV.png
│   ├── training_curve_replication_ridgeCV.svg
│   ├── training_curve_reproduction.pdf
│   ├── training_curve_reproduction.png
│   └── training_curve_reproduction.svg
├── figure2
│   ├── training_curve_ridge_huth.pdf
│   ├── training_curve_ridge_huth.png
│   ├── training_curve_ridge_huth.svg
│   ├── training_curve_ridgeCV.pdf
│   ├── training_curve_ridgeCV.png
│   └── training_curve_ridgeCV.svg
└── figure3
    ├── colorbar.pdf
    ├── colorbar.png
    ├── colorbar.svg
    ├── semantic_performance_extension_ridgeCV.pdf
    ├── semantic_performance_extension_ridgeCV.png
    ├── semantic_performance_extension_ridgeCV.svg
    ├── training_curve_extension_ridgeCV.pdf
    ├── training_curve_extension_ridgeCV.png
    └── training_curve_extension_ridgeCV.svg
```

## Reproduce correlation results

1. Follow step 1. and 2. from above.

2. Download data

```sh
# all stories for the 3 subjects in our analysis
python src/encoders/download_data.py --stories all --subjects UTS01 UTS02 UTS03
```

3. Run test regression

```sh
python src/encoders/run_all.py\
  --cross_validation simple\
  --subject UTS02\
  --feature eng1000\
  --n_train_stories 2\
  --n_repeats 3\
  --ridge_implementation ridgeCV\
  --run_folder_name example
```

4. Run regressions reproducing our results

```sh
# Reproduction
python -m lebel_encoding.run_all_replication \
--subject UTS01 UTS02 UTS03 \
--feature eng1000 \
--test_story wheretheressmoke \
--n_train_stories 1 2 3 5 7 9 11 13 15 17 19 21 23 25 \
--n_repeats 15 \
--trim 5 \
--ndelays 4 \
--nboots 20 \
--chunklen 10 \
--nchunks 10 \
--use_corr \
--run_folder_name reproduction


# Replication ridgeCV
python -m encoders.run_all \
--subject UTS01 UTS02 UTS03 \
--feature eng1000 \
--ndelays 4 \
--interpolation 'lanczos' \
--ridge_implementation 'ridgeCV' \
--n_train_stories 1 2 3 5 7 9 11 13 15 17 19 21 23 25 \
--test_story 'wheretheressmoke' \
--cross_validation 'simple' \
--n_repeats 15 \
--no_keep_train_stories_in_mem \
--run_folder_name replication_ridgeCV


# Replication ridge_huth
python -m encoders.run_all \
--subject UTS01 UTS02 UTS03 \
--feature eng1000 \
--n_train_stories 1 2 3 5 7 9 11 13 15 17 19 21 23 25  \
--test_story wheretheressmoke \
--cross_validation 'simple' \
--interpolation 'lanczos' \
--ridge_implementation 'ridge_huth' \
--n_repeats 15 \
--ndelays 4 \
--nboots 20 \
--chunklen 10 \
--nchunks 10 \
--use_corr \
--no_keep_train_stories_in_mem \
--run_folder_name replication_ridge_huth


# Extension
python -m encoders.run_all \
--subject UTS01 UTS02 UTS03 \
--feature 'envelope' \
--do_shuffle \
--ndelays 4 \
--interpolation 'lanczos' \
--ridge_implementation 'ridgeCV' \
--n_train_stories 1 2 3 5 7 9 11 13 15 17 19 21 23 25 \
--test_story 'wheretheressmoke' \
--cross_validation 'simple' \
--n_repeats 15 \
--no_keep_train_stories_in_mem \
--run_folder_name extension_ridgeCV
```

**It is likely you will need to run the analyses on a HPC system due to RAM requirements.**

For examples how we deployed the scripts on a cluster, see the [hpc](hpc) folder.

## Development setup

1. [Install poetry](https://python-poetry.org/docs/#installation)
2. Tested for `poetry==2.1.2`
3. Run following commands:

```sh
# setup up conda environment (optional)
conda create -n enc python=3.12
conda activate enc

# install dependencies
poetry install

# install pre-commit
pre-commit install

# download some data for testing
python src/encoders/download_data.py # subject 2 & few stories

# Setup the config with the editor of your choice
nano config.yaml
```

## Team

- [Gabriel Kressing Palacios](https://gabrielkp.com/)
- Gio Li
- [Kristijan Armeni](https://www.kristijanarmeni.net/)

## License
