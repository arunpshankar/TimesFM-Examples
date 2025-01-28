from google.cloud.aiplatform.prediction.predictor import Predictor
from src.config.logging import logger
from google.cloud import aiplatform
import matplotlib.pyplot as plt
from src.config.setup import * 
from typing import Optional
from typing import List
from typing import Dict 
from typing import Any 
import numpy as np
import yaml 


def load_endpoints(file_path: str) -> List[str]:
    """
    Load endpoints from a YAML configuration file.
    
    Args:
        file_path (str): Path to the YAML file.
        
    Returns:
        List[str]: A list of endpoint resource names.
    """
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return data.get("endpoints", [])
    except Exception as e:
        logger.error("Failed to load endpoints from %s: %s", file_path, e)
        raise


def create_vertex_ai_predictor(endpoint_name: str) -> Predictor:
    """
    Create a Vertex AI predictor for the given endpoint.
    
    Args:
        endpoint_name (str): The fully qualified endpoint resource name.
        
    Returns:
        Predictor: Vertex AI Predictor instance.
    """
    try:
        predictor = aiplatform.Endpoint(endpoint_name=endpoint_name)
        logger.info("Successfully created Vertex AI Predictor for endpoint: %s", endpoint_name)
        return predictor
    except Exception as e:
        logger.error("Failed to create Vertex AI Predictor: %s", e)
        raise


def make_inference(predictor: Predictor, instances: List[Dict[str, Any]]) -> Any:
    """
    Make inference using the Vertex AI endpoint.
    
    Args:
        predictor (Predictor): The Vertex AI Predictor instance.
        instances (List[Dict[str, Any]]): List of input instances for prediction.
        
    Returns:
        Any: Prediction results.
    """
    try:
        results = predictor.predict(instances=instances)
        logger.info("Inference completed successfully.")
        return results
    except Exception as e:
        logger.error("Failed to make inference: %s", e)
        raise


class Visualizer:
    def __init__(self, nrows: int, ncols: int):
        """
        Initialize the Visualizer with a grid of subplots.

        Args:
            nrows (int): Number of rows of subplots.
            ncols (int): Number of columns of subplots.
        """
        self.ncols = ncols
        self.num_images = nrows * ncols
        self.fig, self.axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 4, nrows * 2.5)
        )

        # Ensure self.axes is always iterable
        if isinstance(self.axes, np.ndarray):
            self.axes = self.axes.flatten()
        else:
            self.axes = [self.axes]  # Wrap single Axes object in a list

        self.index = 0

    def visualize_forecast(
        self,
        context: List[float],
        horizon_mean: List[float],
        ground_truth: Optional[List[float]] = None,
        horizon_lower: Optional[List[float]] = None,
        horizon_upper: Optional[List[float]] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Visualize the forecast on a subplot.

        Args:
            context (List[float]): Historical context values.
            horizon_mean (List[float]): Forecasted mean values.
            ground_truth (Optional[List[float]]): Actual values (if available).
            horizon_lower (Optional[List[float]]): Lower bound of forecast (if available).
            horizon_upper (Optional[List[float]]): Upper bound of forecast (if available).
            ylabel (Optional[str]): Label for the y-axis.
            title (Optional[str]): Title for the subplot.
        """
        if self.index >= len(self.axes):
            raise ValueError("More visualizations requested than available subplots.")

        # Prepare x-axis range
        plt_range = list(range(len(context) + len(horizon_mean)))

        # Plot context
        self.axes[self.index].plot(
            plt_range,
            context + [np.nan] * len(horizon_mean),
            color="tab:cyan",
            label="Context",
        )

        # Plot forecast
        self.axes[self.index].plot(
            plt_range,
            [np.nan] * len(context) + horizon_mean,
            color="tab:red",
            label="Forecast",
        )

        # Plot ground truth if available
        if ground_truth:
            self.axes[self.index].plot(
                plt_range,
                [np.nan] * len(context) + ground_truth,
                color="tab:purple",
                label="Ground Truth",
            )

        # Plot forecast bounds if available
        if horizon_upper and horizon_lower:
            self.axes[self.index].plot(
                plt_range,
                [np.nan] * len(context) + horizon_upper,
                color="tab:orange",
                linestyle="--",
                label="Forecast Upper",
            )
            self.axes[self.index].plot(
                plt_range,
                [np.nan] * len(context) + horizon_lower,
                color="tab:orange",
                linestyle=":",
                label="Forecast Lower",
            )
            self.axes[self.index].fill_between(
                plt_range,
                [np.nan] * len(context) + horizon_upper,
                [np.nan] * len(context) + horizon_lower,
                color="tab:orange",
                alpha=0.2,
            )

        # Add labels and title
        if ylabel:
            self.axes[self.index].set_ylabel(ylabel)
        if title:
            self.axes[self.index].set_title(title)

        # Finalize subplot
        self.axes[self.index].set_xlabel("Time")
        self.axes[self.index].legend()
        self.index += 1

    def save(self, path: str) -> None:
        """
        Save all subplots to a file.

        Args:
            path (str): Path to save the figure.
        """
        self.fig.tight_layout()
        self.fig.savefig(path, dpi=300)
        plt.close(self.fig)