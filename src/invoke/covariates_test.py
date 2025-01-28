from src.invoke.helper import create_vertex_ai_predictor
from src.invoke.helper import make_inference
from src.invoke.helper import load_endpoints
from src.config.logging import logger
from collections import defaultdict
import pandas as pd
import json
import os


# https://github.com/google-research/timesfm/blob/master/notebooks/covariates.ipynb

def get_endpoint_name(yaml_file_path: str) -> str:
    """
    Loads endpoints from the given YAML configuration and returns the first one.
    
    :param yaml_file_path: Path to the YAML file containing endpoints.
    :return: The name of the first endpoint found.
    :raises ValueError: If no endpoints are found.
    """
    endpoints = load_endpoints(yaml_file_path)
    if not endpoints:
        logger.error("No endpoints found in the configuration file.")
        raise ValueError("Endpoints configuration is empty.")
    logger.info("Using endpoint: %s", endpoints[0])
    return endpoints[0]


def load_electricity_data(file_path: str) -> pd.DataFrame:
    """
    Loads electricity price forecasting data from a CSV file.

    :param file_path: Path to the CSV file containing electricity data.
    :return: A Pandas DataFrame with the loaded data.
    :raises FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        logger.error("Electricity data file not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("Loading electricity data from: %s", file_path)
    return pd.read_csv(file_path)


def get_batched_data_fn(data: pd.DataFrame, batch_size: int = 128, context_len: int = 120, horizon_len: int = 24):
    """
    Prepares batched data for forecasting.

    :param data: A Pandas DataFrame containing electricity data.
    :param batch_size: Number of examples per batch.
    :param context_len: Length of the forecasting context.
    :param horizon_len: Length of the forecasting horizon.
    :return: A generator function yielding batches of data.
    """
    examples = defaultdict(list)
    num_examples = 0

    for country in data["unique_id"].unique():
        sub_df = data[data["unique_id"] == country]
        for start in range(0, len(sub_df) - (context_len + horizon_len), horizon_len):
            num_examples += 1
            examples["country"].append(country)
            examples["inputs"].append(
                sub_df["y"][start:(context_end := start + context_len)].tolist()
            )
            examples["gen_forecast"].append(
                sub_df["gen_forecast"][start:context_end + horizon_len].tolist()
            )
            examples["week_day"].append(
                sub_df["week_day"][start:context_end + horizon_len].tolist()
            )
            examples["timestamps"].append(
                sub_df["ds"][start:context_end].tolist()  # Use `ds` for timestamps
            )
            examples["outputs"].append(
                sub_df["y"][context_end:(context_end + horizon_len)].tolist()
            )

    def data_fn():
        for i in range(1 + (num_examples - 1) // batch_size):
            yield {k: v[(i * batch_size):((i + 1) * batch_size)] for k, v in examples.items()}

    return data_fn


def perform_forecast_with_and_without_covariates(predictor, data_fn, output_dir, context_len, horizon_len):
    """
    Performs forecasting with and without covariates and saves the results to JSON files.

    :param predictor: Vertex AI predictor instance.
    :param data_fn: Batched data generator.
    :param output_dir: Directory to save the forecast results.
    :param context_len: Length of the forecasting context.
    :param horizon_len: Length of the forecasting horizon.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, example in enumerate(data_fn()):
        # Convert timestamps in `ds` to ISO 8601 format
        iso_timestamps = [
            [pd.Timestamp(ts).isoformat() for ts in timestamp_list]
            for timestamp_list in example["timestamps"]
        ]

        # Prepare inference payloads
        payload_without_covariates = [
            {
                "input": example["inputs"][j],
                "horizon": horizon_len,
                "timestamp": iso_timestamps[j],
            }
            for j in range(len(example["inputs"]))
        ]

        payload_with_covariates = [
            {
                "input": example["inputs"][j],
                "horizon": horizon_len,
                "timestamp": iso_timestamps[j],
                "dynamic_numerical_covariates": {"gen_forecast": example["gen_forecast"][j]},
                "dynamic_categorical_covariates": {"week_day": example["week_day"][j]},
                "static_categorical_covariates": {"country": example["country"][j]},
            }
            for j in range(len(example["inputs"]))
        ]

        # Make predictions
        raw_forecast = make_inference(predictor, payload_without_covariates)
        cov_forecast = make_inference(predictor, payload_with_covariates)

        # Save forecasts to JSON files
        raw_forecast_file = os.path.join(output_dir, f"raw_forecast_batch_{i + 1}.json")
        cov_forecast_file = os.path.join(output_dir, f"cov_forecast_batch_{i + 1}.json")

        with open(raw_forecast_file, "w") as raw_file:
            json.dump(raw_forecast, raw_file, indent=4)

        with open(cov_forecast_file, "w") as cov_file:
            json.dump(cov_forecast, cov_file, indent=4)


def test():
    """
    Main entry point for the script. Loads data, prepares instances, performs forecasting,
    and saves the results to JSON files.
    """
    # 1. Load electricity data
    csv_file_path = "./data/input/electricity.csv"
    data = load_electricity_data(csv_file_path)

    # 2. Prepare batched data
    batch_size = 128
    context_len = 120
    horizon_len = 24
    data_fn = get_batched_data_fn(data, batch_size, context_len, horizon_len)

    # 3. Initialize Vertex AI Predictor
    yaml_file_path = "./config/endpoints.yml"
    endpoint_name = get_endpoint_name(yaml_file_path)
    predictor = create_vertex_ai_predictor(endpoint_name)

    # 4. Perform forecasting with and without covariates
    output_dir = "./data/output/forecasts"
    perform_forecast_with_and_without_covariates(
        predictor, data_fn, output_dir, context_len, horizon_len
    )


if __name__ == "__main__":
    test()
