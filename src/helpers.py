from collections import namedtuple
import contextlib
import logging
import pickle
import timeit
import os
import random
import string
import datetime
import re
from typing import Iterable, List, Mapping
from typing import Any

import torch

@contextlib.contextmanager
def time_activity(activity_name: str):
    """Times the execution of a piece of code
    Args:
        activity_name (str): string that describes the name of an activity. It can be arbitrary
    """
    logging.info("[Timing] %s started", activity_name)
    start = timeit.default_timer()
    yield
    duration = timeit.default_timer() - start
    logging.info("[Timing] %s finished (Took %.2fs)", activity_name, duration)
    
    
def get_chunks(data: List[Any], chunck_num: int) -> Iterable[List[Any]]:
    """
    Divide list of elements into chuncks of `chunk_num` elements each
    except the last chunk if the the total number of elements is not
    divisible by `chunk_num`
    Args:
        data (List[Any]): list of elements
        chunck_num (int): number of elements each chuck should contain
    Returns:
        Iterable[List[Any]]: generator chunks
    """
    for i in range(0, len(data), chunck_num):
        yield data[i : i + chunck_num]
    

def get_dir(*paths) -> str:
    """Creates a dir from a list of directories (like os.path.join), runs os.makedirs and returns the name
    Args:
        paths (List[str]): list of string that constitutes the path
    Returns:
        str: the created or existing path
    """
    directory = os.path.join(*paths)
    os.makedirs(directory, exist_ok=True)
    return directory

def log(dir: str, data: Mapping[str, Any], step: int):
    """Logs anython to a pickle file
    Args:
        dir (str): data directory of the experiment
        key (str): key for the logged data
        data (Any): data to be logged
    """
    filepath = os.path.join(dir, "logs.pkl")
    
    # create the pickle file if it doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, "wb") as f:
            pickle.dump({step: data}, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:   
        # load the pickle file
        with open(filepath, "rb") as f:
            logs = pickle.load(f)
            logs[step] = {
                **logs.get(step, {}),
                **data
            }
        
        with open(filepath, "wb") as f:
            pickle.dump(logs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
def initialize_logging(log_filepath):
    # remove previous log file
    if os.path.exists(log_filepath):
        os.unlink(log_filepath)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    
def save_pytorch_things(dir, things):
    torch.save(things, os.path.join(dir, "model.pth"))
    return os.path.join(dir, "model.pth")


def generate_random_string(length: int = 10):
    """
    generate random alphanumeric characters
    Args:
        legnth (int): length of the string to be generated
    """
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def get_current_date() -> str:
    """
    returns the current date in format => 20XX-12-30
    """
    return datetime.datetime.now().strftime("%Y-%m-%d")


def get_current_time() -> str:
    """
    returns the current time in format => 24-60-60
    """
    return datetime.datetime.now().strftime("%H-%M-%S")


def get_new_run_dir_params():
    # hydra uses timestamp to create logs and
    # stuffs. This might be a problem when running
    # simultaneous process. this ensures that no
    # hydra runs share the same folders
    d_suffix = os.path.join(
        get_current_date(), get_current_time(), generate_random_string()
    )
    run_dir = os.path.join("results", d_suffix)
    multirun_dir = os.path.join("results/multirun", d_suffix)
    return {"hydra.run.dir": run_dir, "hydra.sweep.dir": multirun_dir}


def is_path_creatable(dir: str):
    """check is the directory is creatable
    Args:
        dir (str): directory
    Returns:
        bool: check result
    """
    valid_dir_pattern = re.compile(
        "^((\.|\.\.)(\/)?)?((.+)\/([^\/]+))?(\/)?(.+)?$"  # noqa: W605 escapes are fine flake!
    )
    has_no_space = " " not in dir
    not_parent_of_cwd = (
        not dir.startswith("/") and not dir.startswith("..")
    ) or dir.startswith(os.getcwd())
    return has_no_space and bool(re.match(valid_dir_pattern, dir)) and not_parent_of_cwd


def has_valid_hydra_dir_params(arguments: List[str]):
    """
    check list of arguments has hydra run dir
    Args:
        arguments (List[str]): list of arguments
    """

    def is_valid_param(argument: str, param: str):
        """
        check the argument is in the forman "{param}={valid path}"
        Args:
            argument (str): potential argument
            param (str): param key
        Returns:
            bool: result of the check
        """
        pair = argument.split("=")
        return len(pair) == 2 and pair[0] == param and is_path_creatable(pair[1])

    has_hyda_run_dir = any([is_valid_param(arg, "hydra.run.dir") for arg in arguments])
    has_hydra_sweep_dir_param = any(
        [is_valid_param(arg, "hydra.sweep.dir") for arg in arguments]
    )

    return has_hyda_run_dir and has_hydra_sweep_dir_param