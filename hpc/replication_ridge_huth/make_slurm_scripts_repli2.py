from itertools import product
from pathlib import Path

from encoders.utils import check_make_dirs, create_run_folder_name

SUBJECTS = ["UTS01", "UTS02", "UTS03"]
N_STORIES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]


def create_scripts_replication2(
    email: str = None, log_path: str = None, run_folder_name: str = None
) -> None:
    """
    Create scripts as individual jobs for each tranining set size.

    """

    nstories2mem = {
        1: "60GB",
        3: "80GB",
        5: "80GB",
        7: "80GB",
        9: "80GB",
        11: "100GB",
        13: "150GB",
        15: "150GB",
        17: "150GB",
        19: "160GB",
        21: "180GB",
        23: "180GB",
        25: "180GB",
    }

    nstories2time = {
        1: "03:00:00",
        3: "03:00:00",
        5: "03:00:00",
        7: "03:00:00",
        9: "03:00:00",
        11: "03:00:00",
        13: "06:00:00",
        15: "06:00:00",
        17: "06:00:00",
        19: "06:00:00",
        21: "06:00:00",
        23: "06:00:00",
        25: "07:00:00",
    }

    master_script_name = "submit_jobs.sh"
    master_script_f = open(master_script_name, "w")

    if run_folder_name is None:
        run_folder_name = create_run_folder_name()

    combinations = list(product(SUBJECTS, N_STORIES))

    for tup in combinations:
        subject, n_stories = tup

        script_name = f"repli-exp-2_{subject}-{n_stories}.script"

        script_f = open(script_name, "w")

        script_f.write("#!/bin/bash\n")
        script_f.write(f"#SBATCH --job-name={script_name}\n")
        script_f.write(f"#SBATCH --time={nstories2time[n_stories]}\n")
        script_f.write(f"#SBATCH --mem {nstories2mem[n_stories]}\n")
        script_f.write("#SBATCH --partition=parallel\n")
        script_f.write("#SBATCH --signal=USR2\n")
        script_f.write("#SBATCH --nodes=1\n")
        script_f.write("#SBATCH --cpus-per-task=8\n")

        if email:
            script_f.write("#SBATCH --mail-type=END,FAIL\n")
            script_f.write(f"#SBATCH --mail-user={email}\n")

        if log_path:
            script_f.write(f"#SBATCH --output={log_path}/{script_name}.job.%j.out\n")
            script_f.write(f"#SBATCH --error={log_path}/{script_name}.job.%j.err\n\n")

        script_f.write("ml anaconda3/2024.02-1\n")
        script_f.write("conda activate enc\n\n")

        script_f.write("python -m encoders.run_all \\\n")
        script_f.write(f"--subject {subject} \\\n")
        script_f.write("--feature 'all' \\\n")
        script_f.write(f"--n_train_stories {n_stories} \\\n")
        script_f.write("--test_story wheretheressmoke \\\n")
        script_f.write("--cross_validation 'simple' \\\n")
        script_f.write("--interpolation 'lanczos' \\\n")
        script_f.write("--ridge_implementation 'ridge_huth' \\\n")
        script_f.write("--n_repeats 15 \\\n")
        script_f.write("--ndelays 4 \\\n")
        script_f.write("--nboots 20 \\\n")
        script_f.write("--chunklen 10 \\\n")
        script_f.write("--nchunks 10 \\\n")
        script_f.write("--use_corr \\\n")
        script_f.write("--no_keep_train_stories_in_mem \\\n")
        script_f.write(f"--run_folder_name {run_folder_name} \\\n")

        script_f.close()
        print(f"Created {script_name}")

        master_script_f.write(f"sbatch {script_name}\n")

    master_script_f.close()
    print(f"Created {master_script_name}")

    return None


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--email", type=str, help="Email address to send the SLURM emails to"
    )
    parser.add_argument(
        "--log_path", type=str, help="Path to wrinte slurm .err and .out files"
    )
    parser.add_argument(
        "--run_folder_name",
        type=str,
        help="Name of the run folder, if not specified "
        "it is created via create_run_folder_name()",
    )

    args = parser.parse_args()

    if args.log_path:
        Path(args.log_path).mkdir(parents=True, exist_ok=True)

    create_scripts_replication2(
        email=args.email,
        log_path=args.log_path,
        run_folder_name=args.run_folder_name,
    )


if __name__ == "__main__":
    main()
