import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from random import choices
from typing import List, Optional, Union

import numpy as np
import yaml

FORMAT = "[%(levelname)s] %(name)s.%(funcName)s - %(message)s"

logging.basicConfig(format=FORMAT)


def get_logger(
    name=__name__,
    log_level=logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Initializes command line logger."""

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if log_file is not None:
        formatter = logging.Formatter(FORMAT)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


log = get_logger(__name__)


def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config


def check_make_dirs(
    paths: Union[
        str,
        Path,
        List[Union[str, Path]],
    ],
    verbose: bool = True,
    isdir: bool = False,
) -> None:
    """Create base directories for given paths if they do not exist.

    Parameters
    ----------
    paths: List[str] | str
        A path or list of paths for which to check the basedirectories
    verbose: bool, default=True
        Whether to log the output path
    isdir: bool, default=False
        Treats given path(s) as diretory instead of only checking the basedir.
    """

    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        if isdir and (path != "") and (not os.path.exists(path)):
            os.makedirs(path)
        elif os.path.dirname(path) != "" and not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if verbose:
            log.info(f"Output path: {path}")


def create_run_folder_name(seed=None) -> str:
    """Returns the name of the run folder including
    a random id and the current date."""

    if seed:
        random.seed(seed)

    date = datetime.today().strftime("%Y-%m-%d_%H-%M")
    rand_num = "".join(random.choices("0123456789", k=6))
    return f"{date}_{rand_num}"


def counter(iterable, countevery=100, total=None, logger=logging.getLogger("counter")):
    """Logs a status and timing update to [logger] every [countevery] draws from
    [iterable]. If [total] is given, log messages will include the estimated time
    remaining.
    """
    start_time = time.time()

    ## Check if the iterable has a __len__ function, use it
    ##  if no total length is supplied
    if total is None:
        if hasattr(iterable, "__len__"):
            total = len(iterable)

    for count, thing in enumerate(iterable):
        yield thing

        if not count % countevery:
            current_time = time.time()
            rate = float(count + 1) / (current_time - start_time)

            if rate > 1:  ## more than 1 item/second
                ratestr = "%0.2f items/second" % rate
            else:  ## less than 1 item/second
                ratestr = "%0.2f seconds/item" % (rate**-1)

            if total is not None:
                remitems = total - (count + 1)
                remtime = remitems / rate
                timestr = ", %s remaining" % time.strftime(
                    "%H:%M:%S", time.gmtime(remtime)
                )
                itemstr = "%d/%d" % (count + 1, total)
            else:
                timestr = ""
                itemstr = "%d" % (count + 1)

            formatted_str = "%s items complete (%s%s)" % (itemstr, ratestr, timestr)
            if logger is None:
                print(formatted_str)
            else:
                logger.info(formatted_str)
