from src.invoke.helper import create_vertex_ai_predictor
from src.utils.inference import get_endpoint_name
from src.invoke.helper import make_inference
from src.invoke.helper import Visualizer
from src.config.logging import logger
import os
import pandas as pd


def load_temperature_data(file_path: str) -> pd.DataFrame:
    """
    Loads temperature data from a CSV file.

    :param file_path: Path to the CSV file containing temperature data.
    :return: A Pandas DataFrame with the loaded data.
    :raises FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        logger.error("Temperature data file not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info("Loading temperature data from: %s", file_path)
    return pd.read_csv(file_path)


def prepare_instances(data: pd.DataFrame) -> tuple:
    """
    Prepares input data, timestamps, and ground truths for prediction and visualization.

    :param data: A Pandas DataFrame containing temperature and date columns.
    :return: A tuple of inputs, timestamps, and ground truths.
    """
    temperature = data["meantemp"].to_list()
    dates = data["date"].to_list()

    inputs = [
        temperature[0:200],
        temperature[300:600],
        temperature[700:1200],
    ]
    timestamps = [
        dates[0:200],
        dates[300:600],
        dates[700:1200],
    ]
    ground_truths = [
        temperature[200:300],
        temperature[600:700],
        temperature[1200:1300],
    ]

    logger.info("Prepared instances, timestamps, and ground truths for prediction.")
    return inputs, timestamps, ground_truths


def perform_inference(predictor, inputs, timestamps, horizon=100) -> list:
    """
    Performs inference using the predictor and input data.

    :param predictor: A predictor instance for making predictions.
    :param inputs: A list of input data.
    :param timestamps: A list of timestamps corresponding to the inputs.
    :param horizon: The prediction horizon.
    :return: A list of predictions.
    """
    instances = [
        {
            "input": each_input,
            "horizon": horizon,
            "timestamp": each_timestamp,
            "timestamp_format": "%Y-%m-%d",
        }
        for each_input, each_timestamp in zip(inputs, timestamps)
    ]

    logger.info("Performing inference with %d instances.", len(instances))
    return make_inference(predictor, instances)


def save_visualizations(viz, save_dir: str, file_prefix: str = "forecast") -> None:
    """
    Saves visualizations to the specified directory.

    :param viz: A Visualizer instance containing the plots.
    :param save_dir: Directory where the visualizations will be saved.
    :param file_prefix: Prefix for the saved file names.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info("Created directory: %s", save_dir)

    save_path = os.path.join(save_dir, f"{file_prefix}.png")
    viz.save(save_path)
    logger.info("Visualization saved to: %s", save_path)


def test():
    """
    Main entry point for the script. Loads data, prepares instances, performs inference,
    and visualizes the results.
    """
    # 1. Load temperature data
    csv_file_path = "./data/input/temperatures.csv"
    data = load_temperature_data(csv_file_path)

    # 2. Prepare instances
    inputs, timestamps, ground_truths = prepare_instances(data)

    # 3. Load the endpoint name
    yaml_file_path = "./config/endpoints.yml"
    endpoint_name = get_endpoint_name(yaml_file_path)

    # 4. Create the Vertex AI Predictor
    predictor = create_vertex_ai_predictor(endpoint_name)

    # 5. Perform inference
    predictions = perform_inference(predictor, inputs, timestamps)
    logger.info("Predictions: %s", predictions)

    # 6. Visualization
    viz = Visualizer(nrows=1, ncols=3)

    for task_i in range(len(inputs)):
        viz.visualize_forecast(
            inputs[task_i],
            predictions[0][task_i]["point_forecast"][:100],
            ground_truth=ground_truths[task_i],
            title=f"Daily Temperature in Delhi, India, Task {task_i + 1}",
            ylabel="Temperature (°C)",
        )

    # 7. Save visualizations
    save_dir = "./data/visuals"
    save_visualizations(viz, save_dir, file_prefix="temperature_forecasts")


    # Visualize with Quantiles for Anamoly Detection 
    viz = Visualizer(nrows=1, ncols=3)

    for task_i in range(len(inputs)):
        viz.visualize_forecast(
            inputs[task_i],
            predictions[0][task_i]["point_forecast"],
            ground_truth=ground_truths[task_i],
            horizon_lower=predictions[0][task_i]["p30"],
            horizon_upper=predictions[0][task_i]["p70"],
            title=f"Daily Temperature in Delhi, India, Task {task_i + 1}",
            ylabel="Temperature (°C)",
        )

    save_dir = "./data/visuals"
    save_visualizations(viz, save_dir, file_prefix="temperature_forecasts_anamoly")


if __name__ == "__main__":
    test()
