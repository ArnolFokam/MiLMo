import os
import pickle
from typing import Dict, Tuple
from matplotlib import pyplot as plt

import wandb
from omegaconf import DictConfig

class MetricsLogger:
    """
    A class for logging metrics during training.
    """
    
    POINT: str = "point"
    IMAGE: str = "image"
    BAR: str = "bar"
    TABLE: str = "table"
    MATPLOTLIB_BAR: str = "matplotlib_barplot"
    
    LOG_FILENAME: str = "logs.pkl"
    
    DOWNSTREAM_EVAL_PREFIX="downstream_eval"
    
    def __init__(self, results_dir: str, train_cfg: DictConfig, *args, **kwargs) -> None:
        """
        Initializes the MetricsLogger.

        Args:
            results_dir (str): Directory where logs will be saved.
            train_cfg (DictConfig): Training configuration.
        """
        self.train_cfg = train_cfg
        self.results_dir = results_dir
        self.logs = {}

    def log(self, metrics: Dict[str, Tuple[str, Tuple]], step: int):
        """
        Logs metrics at a specific training step.

        Args:
            metrics (Dict[str, Tuple[str, Tuple]]): Dictionary of metrics to be logged.
            step (int): Training step.

        Returns:
            None
        """
        # Update or add metrics at the specified step
        self.logs[step] = {
            **self.logs.get(step, {}),
            **metrics
        }

    def state_dict(self):
        """
        Returns the current state of the MetricsLogger.

        Returns:
            dict: Current state containing logs.
        """
        return {
            "logs": self.logs
        }

    def load_state_dict(self, state):
        """
        Loads the state of the MetricsLogger.

        Args:
            state (dict): State to be loaded.

        Returns:
            None
        """
        # Load the logs from the provided state
        self.logs = state["logs"]

    def save_logs(self):
        """
        Saves logs to a file in the specified results directory.

        Returns:
            None
        """
        # Create the full filepath
        filepath = os.path.join(self.results_dir, self.LOG_FILENAME)

        # Save the logs to a file using pickle
        with open(filepath, "wb") as f:
            pickle.dump(self.logs, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    @staticmethod
    def map_type_to_wandb_type(type: str, params: Tuple) -> Tuple:
        """
        Maps a type to a type that can be logged by wandb.

        Args:
            type (str): Type of metric.
            params (Tuple): Parameters of the metric.

        Returns:
            Tuple: Type and parameters of the metric.
        """
        if type == MetricsLogger.POINT:
            return round(float(params[0]), 3)
        elif type == MetricsLogger.IMAGE:
            return wandb.Image(params[0], caption=params[1])
        elif type == MetricsLogger.BAR:
            table = wandb.Table(data=params[0], columns = [params[1], params[2]])
            return wandb.plot.bar(
                table, 
                label=params[1], 
                value=params[2],
                title=f"{params[2]} over {params[1]}")
        elif type == MetricsLogger.TABLE:
            return wandb.Table(data=params[0], columns = [params[1], params[2]])
        elif type == MetricsLogger.MATPLOTLIB_BAR:
            # create plot
            plt.bar(params[0], params[1])
            plt.xticks(rotation=25)
            plt.ylim([0.0, 1.0])
            
            # make image from plot
            image = wandb.Image(plt)
            plt.clf()
            
            return image
        else:
            raise ValueError(f"Unknown type {type}")