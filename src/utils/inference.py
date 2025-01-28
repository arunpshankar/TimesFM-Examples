from src.invoke.helper import load_endpoints
from src.config.logging import logger


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