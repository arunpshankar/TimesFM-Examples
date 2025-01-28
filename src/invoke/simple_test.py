
from src.invoke.helper import create_vertex_ai_predictor
from src.invoke.helper import make_inference
from src.invoke.helper import load_endpoints
from src.invoke.helper import Visualizer
from src.config.logging import logger
from src.config.setup import * 
import numpy as np


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


def get_instances() -> list:
    """
    Prepares a list of instances to be sent for prediction.
    
    :return: A list of dictionaries containing the input data.
    """
    # Example data: two sine waves of different frequencies
    return [
        {"input": np.sin(np.linspace(0, 20, 100)).tolist()},
        {"input": np.sin(np.linspace(0, 40, 500)).tolist()},
    ]


def test():
    """
    Main entry point for the script. Loads configuration, creates a predictor,
    performs inference, and visualizes the results.
    """
    # 1. Load the endpoint name
    yaml_file_path = "./config/endpoints.yml"
    endpoint_name = get_endpoint_name(yaml_file_path)

    # 2. Create the Vertex AI Predictor
    predictor = create_vertex_ai_predictor(endpoint_name)
    
    # 3. Prepare instances
    instances = get_instances()
    logger.info("Prepared instances for inference.")

    # 4. Perform inference
    predictions = make_inference(predictor, instances)
    logger.info("Predictions: %s", predictions)

    # 5. Visualization
    viz = Visualizer(nrows=1, ncols=2)
    viz.visualize_forecast(
        instances[0]["input"], predictions[0][0]["point_forecast"], title="Sinusoidal 1"
    )
    viz.visualize_forecast(
        instances[1]["input"], predictions[0][1]["point_forecast"], title="Sinusoidal 2"
    )

    # Save the visualization
    save_path = "./data/visuals/sinusoidal_forecasts.png"
    save_dir = os.path.dirname(save_path)

    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info("Created directory: %s", save_dir)

    viz.save(save_path)
    logger.info("Visualization saved to: %s", save_path)


if __name__ == "__main__":
    test()
