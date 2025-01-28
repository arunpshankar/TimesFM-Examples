from src.invoke.helper import create_vertex_ai_predictor
from src.invoke.helper import make_inference
from src.invoke.helper import load_endpoints
from src.config.logging import logger
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
from typing import Tuple
from typing import List
import pandas as pd
import json
import os


def get_endpoint_name(yaml_file_path: str) -> str:
    """
    Retrieve the first endpoint name from a YAML configuration file.

    Args:
        yaml_file_path (str): Path to the YAML file containing endpoints.

    Returns:
        str: The name of the first endpoint.

    Raises:
        ValueError: If no endpoints are found in the configuration file.
    """
    endpoints = load_endpoints(yaml_file_path)
    if not endpoints:
        logger.error("No endpoints found in the configuration file.")
        raise ValueError("Endpoints configuration is empty.")
    logger.info("Using endpoint: %s", endpoints[0])
    return endpoints[0]


def load_stock_data(file_path: str) -> pd.DataFrame:
    """
    Load stock data from a JSON file into a Pandas DataFrame.

    Args:
        file_path (str): Path to the JSON file containing stock data.

    Returns:
        pd.DataFrame: DataFrame containing the stock data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(file_path):
        logger.error("Stock data file not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("Loading stock data from: %s", file_path)
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def convert_timestamps(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestamp strings in the DataFrame to Python datetime objects.

    Args:
        data (pd.DataFrame): DataFrame containing a 'date' column with timestamp strings.

    Returns:
        pd.DataFrame: Updated DataFrame with 'date' column converted to datetime.

    Raises:
        ValueError: If timestamp conversion fails due to incorrect format.
    """
    fmt_str = "%b %d %Y, %I:%M %p UTC%z"
    try:
        data["date"] = data["date"].apply(lambda x: datetime.strptime(x, fmt_str))
        logger.info("Converted timestamps to datetime objects.")
    except ValueError as e:
        logger.error("Timestamp conversion failed: %s", e)
        raise ValueError("Invalid timestamp format.") from e
    return data


def prepare_stock_instances(data: pd.DataFrame) -> Tuple[List[List[float]], List[List[datetime]]]:
    """
    Prepare stock price and timestamp data for inference.

    Args:
        data (pd.DataFrame): DataFrame containing 'price' and 'date' columns.

    Returns:
        Tuple[List[List[float]], List[List[datetime]]]:
            - List of lists of stock prices.
            - List of lists of datetime objects.
    """
    prices = data["price"].to_list()
    dates = data["date"].to_list()
    logger.info("Prepared stock instances for inference.")
    return [prices], [dates]


def perform_inference(predictor, inputs: List[List[float]], timestamps: List[List[datetime]], horizon: int = 21) -> list:
    """
    Perform inference using the predictor with specified input data and horizon.

    Args:
        predictor: Vertex AI Predictor instance.
        inputs (List[List[float]]): List of lists containing stock price data.
        timestamps (List[List[datetime]]): List of lists containing datetime objects.
        horizon (int): Number of days for the prediction horizon.

    Returns:
        list: Inference results from the predictor.
    """
    instances = []
    for each_input, each_timestamp in zip(inputs, timestamps):
        stringified_dates = [dt.strftime("%Y-%m-%d") for dt in each_timestamp]
        instances.append({
            "input": each_input,
            "horizon": horizon,
            "timestamp": stringified_dates,
            "timestamp_format": "%Y-%m-%d",
        })

    logger.info("Performing inference with %d instances.", len(instances))
    return make_inference(predictor, instances)


def save_visualizations(fig: plt.Figure, save_dir: str, file_prefix: str = "forecast") -> None:
    """
    Save Matplotlib visualizations to a specified directory.

    Args:
        fig (plt.Figure): Matplotlib figure to save.
        save_dir (str): Directory to save the visualization.
        file_prefix (str): Prefix for the saved file name.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info("Created directory: %s", save_dir)

    save_path = os.path.join(save_dir, f"{file_prefix}.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    logger.info("Visualization saved to: %s", save_path)


def test() -> None:
    """
    Main entry point for the script. Loads data, prepares instances, performs inference, and visualizes the results.
    """
    try:
        # Load stock price data
        json_file_path = "./data/input/1M_market_data.json"
        data = load_stock_data(json_file_path)

        # Convert timestamps to datetime objects
        data = convert_timestamps(data)

        # Prepare instances
        inputs, timestamps = prepare_stock_instances(data)

        # Load the endpoint name
        yaml_file_path = "./config/endpoints.yml"
        endpoint_name = get_endpoint_name(yaml_file_path)

        # Create the Vertex AI Predictor
        predictor = create_vertex_ai_predictor(endpoint_name)

        # Perform inference
        inference_result = perform_inference(predictor, inputs, timestamps)
        logger.info("Inference result: %s", inference_result)

        # Extract predictions
        predictions = inference_result.predictions[0]
        point_forecast = predictions["point_forecast"]
        p10, p20, p50, p80, p90 = predictions["p10"], predictions["p20"], predictions["p50"], predictions["p80"], predictions["p90"]

        # Generate forecast dates
        last_date = timestamps[0][-1]
        forecast_dates = [last_date + timedelta(days=i + 1) for i in range(len(point_forecast))]

        # Visualization
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(timestamps[0], inputs[0], label="Actual Prices", marker="o", linestyle="-", linewidth=2, color="dodgerblue")
        ax.plot(forecast_dates, point_forecast, label="Forecast (Mean)", marker="o", linestyle="--", linewidth=2, color="orange")
        ax.fill_between(forecast_dates, p10, p90, color="lightcoral", alpha=0.3, label="P10-P90 Interval")
        ax.fill_between(forecast_dates, p20, p80, color="gold", alpha=0.5, label="P20-P80 Interval")
        ax.fill_between(forecast_dates, p50, point_forecast, color="limegreen", alpha=0.7, label="P50-Forecast Interval")

        ax.set_title("Stock Price Forecast Nvidia", fontsize=20, fontweight="bold")
        ax.set_xlabel("Date", fontsize=16)
        ax.set_ylabel("Price (USD)", fontsize=16)
        ax.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate(rotation=45)
        ax.legend(fontsize=14, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0)

        # Save visualization
        save_visualizations(fig, "./data/visuals", "stock_price_forecast_nvidia_1m")

    except Exception as e:
        logger.error("An error occurred: %s", e)
        raise


if __name__ == "__main__":
    test()
