from src.config.logging import logger
from typing import Dict, Any
import yaml
import os

SETUP_CONFIG_PATH = './config/setup.yml'
SERVING_CONFIG_PATH = './config/serve.yml'

class _Config:
    """
    Singleton class to manage application configuration for project and models.

    Attributes:
    -----------
    PROJECT_ID : str
        The project ID from the setup configuration file.
    REGION : str
        The region from the setup configuration file.
    CREDENTIALS_PATH : str
        The path to the Google credentials JSON file.
    BUCKET_NAME : str
        The name of the GCS bucket where models and artifacts are stored.
    MODELS : dict
        The models and adapters configuration loaded from models.yml.

    Methods:
    --------
    _load_config(config_path: str) -> Dict[str, Any]:
        Load the YAML configuration from the given path.
    _set_google_credentials(credentials_path: str) -> None:
        Set the Google application credentials environment variable.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Ensure that only one instance of the _Config class is created (Singleton pattern).
        """
        if not cls._instance:
            cls._instance = super(_Config, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True

        # Load setup and models configurations
        self.__setup_config = self._load_config(SETUP_CONFIG_PATH)
        self.__hosting_config = self._load_config(SERVING_CONFIG_PATH)

        # Project-level configuration
        self.PROJECT_ID = self.__setup_config['project_id']
        self.REGION = self.__setup_config['region']
        self.CREDENTIALS_PATH = self.__setup_config['credentials_json']
        self.BUCKET_NAME = self.__setup_config['bucket_name']

        # Serving configuration
        self.MODEL_LOCATION = self.__hosting_config['model_location']
        self.MODEL_NAME = self.__hosting_config['model_name']
        self.SERVE_DOCKER_URI = self.__hosting_config['serve_docker_uri']
        self.SERVICE_ACCOUNT = self.__hosting_config['service_account']
        self.MODEL_DISPLAY_NAME = self.__hosting_config['model_display_name']
        self.MACHINE_TYPE = self.__hosting_config['machine_type']
        self.ACCELERATOR_TYPE = self.__hosting_config['accelerator_type']
        self.ACCELERATOR_COUNT = self.__hosting_config['accelerator_count']
        self.DEPLOY_SOURCE = self.__hosting_config['deploy_source']
        self.DEDICATED_ENDPOINT_ENABLED = self.__hosting_config['use_dedicated_endpoint']
        self.HORIZON = self.__hosting_config['horizon']
        self.TIMESFM_BACKEND = self.__hosting_config['timesfm_backend']
        self.DEPLOY_REQUEST_TIMEOUT = self.__hosting_config['deploy_request_timeout']

        # Set credentials
        self._set_google_credentials(self.CREDENTIALS_PATH)

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """
        Load the YAML configuration from the given path.

        Parameters:
        -----------
        config_path : str
            Path to the YAML configuration file.

        Returns:
        --------
        Dict[str, Any]
            Loaded configuration data.

        Raises:
        -------
        Exception
            If the configuration file fails to load, logs the error.
        """
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load the configuration file at {config_path}. Error: {e}")
            raise

    @staticmethod
    def _set_google_credentials(credentials_path: str) -> None:
        """
        Set the Google application credentials environment variable.

        Parameters:
        -----------
        credentials_path : str
            Path to the Google credentials file.
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path


# Create a single instance of the _Config class.
config = _Config()
