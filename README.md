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

## Reproduce figures

The reproduction of the figures requires the installation of the relevant dependencies and the correlation results of various encoding model runs.
We provide these results in an online repository.
See below (Reproduce correlation results) how to reproduce the correlation results.

1. Clone repository, change directory into repository directory

```sh
git clone git@github.com:GabrielKP/enc.git
cd enc
```

2. Setup virtual environment

```sh
# conda environment
conda create -n enc python=3.12
conda activate enc

# install package
pip install .

# install git-annex (required to download data from Lebel et al. 2011)

```

3. Install [git-annex and datalad](https://handbook.datalad.org/en/latest/intro/installation.html) (required to download data from Lebel et al. 2011)

4. Download repository data

```sh
# Only download data required for plotting results
python src/encoders/download_data.py [-d DATA_DIR] --figures
```

Without `-d DATA_DIR` the data will be downloaded into the folder `ds003020` of the project directory.
To download the data into a custom dir, specify `-d DATA_DIR` (it is recommended to call the last folder `ds003020` as that is the default dataset name).

5. Setup/check `config.yaml`. It should be created automatically by the download script, if not copy it from `config.example.yaml`. The important dir is the `DATA_DIR`.


> [!NOTE]
> If you get the error `ImportError: cannot import name 'getargspec' from 'inspect'` then try to update your datalad version `python -m pip install datalad --upgrade`

6. Download correlation results

Download [`runs.zip` file](https://osf.io/download/g9cy3) and unzip it such that you have a `runs` directory with the experiment folders as subdirectories:

```
runs/extension_ridgeCV
runs/replication_ridge_huth
runs/replication_ridgeCV
runs/reproduction
```

7. Install [inkscape](https://inkscape.org/) (required for plotting):

Open the `config.yaml` and set the following values accordingly.

```yaml
INKSCAPE_PATH: path/to/inkscape/binary
INKSCAPE_VERSION: X.Y.Z
```

For mac, you usually can [find inkscape as described here](https://stackoverflow.com/a/22085247).

8. Configure pycortex (required for plotting):

**Script**

```sh
python src/encoders/update_pycortex_config.py
```

**Manual**

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

9. Reproduce the plots in the figures:

```sh
# will create plots for all figures
python src/encoders/plot.py
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
