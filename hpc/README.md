This folder contains SLURM scripts to deploy 4 experiments using the functions in the `src.encoders` module.

## Subfolders

```
├── extension_ridgeCV       |--> extension experiment
├── reproduction            |--> reproduction experiment
├── replication_ridgeCV     |--> replication experiment
├── replication_ridge_huth  |--> replication experiment (patched code)
└── README.md
```

## Subfolder contents
Each subfolder contains:

1. *.py script used to create *.scripts (SLURM job) and `submit_jobs.sh`
1. *.script files, which are individual SLURM jobs. Each job runs model fits for one training set size and all predictors.
1. `submit_jobs.sh` file a bash script submiting all *.script files to SLURM

### Using *.py scripts

For example for reproduction experiment:
```python
python make_slurm_scripts_repro.py
```

Or you can specify `email`, `log_path`, and `run_folder_name` to be included in SLURM params:
```python
python make_slurm_scripts_repro.py --email "myemail@email.com" --log_path "/path/to/log/folder" --run_folder_name "name_of_results_folder"
```

The script prints an output like this:

```bash
>>> python make_slurm_scripts_repro.py
Created run-all_repro_UTS01-1.script
Created run-all_repro_UTS01-3.script
Created run-all_repro_UTS01-5.script
Created run-all_repro_UTS01-7.script
Created run-all_repro_UTS01-9.script
Created run-all_repro_UTS01-11.script
Created run-all_repro_UTS01-13.script
Created run-all_repro_UTS01-15.script
Created run-all_repro_UTS01-17.script
Created run-all_repro_UTS01-19.script
Created run-all_repro_UTS01-21.script
Created run-all_repro_UTS01-23.script
Created run-all_repro_UTS01-25.script
Created run-all_repro_UTS02-1.script
Created run-all_repro_UTS02-3.script
Created run-all_repro_UTS02-5.script
Created run-all_repro_UTS02-7.script
Created run-all_repro_UTS02-9.script
Created run-all_repro_UTS02-11.script
Created run-all_repro_UTS02-13.script
Created run-all_repro_UTS02-15.script
Created run-all_repro_UTS02-17.script
Created run-all_repro_UTS02-19.script
Created run-all_repro_UTS02-21.script
Created run-all_repro_UTS02-23.script
Created run-all_repro_UTS02-25.script
Created run-all_repro_UTS03-1.script
Created run-all_repro_UTS03-3.script
Created run-all_repro_UTS03-5.script
Created run-all_repro_UTS03-7.script
Created run-all_repro_UTS03-9.script
Created run-all_repro_UTS03-11.script
Created run-all_repro_UTS03-13.script
Created run-all_repro_UTS03-15.script
Created run-all_repro_UTS03-17.script
Created run-all_repro_UTS03-19.script
Created run-all_repro_UTS03-21.script
Created run-all_repro_UTS03-23.script
Created run-all_repro_UTS03-25.script
Created submit_jobs.sh
```
