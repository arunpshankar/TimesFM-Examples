from google.cloud.aiplatform.initializer import global_config
from google.cloud.aiplatform import Endpoint
from google.cloud.aiplatform import Model 
from src.config.setup import config
from src.config.setup import logger
from datetime import datetime
import yaml
import os


def create_endpoint() -> Endpoint:
    """
    Creates an endpoint in Google AI Platform with a specified model name.

    Args:
        model_name (str): Name of the model for which the endpoint is being created.

    Returns:
        Endpoint: The created endpoint object.
    """
    try:
        # Generate a unique display name using the model name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        endpoint_name = f"{config.MODEL_DISPLAY_NAME}-{timestamp}-endpoint"

        logger.info(f"Creating endpoint with display name: {endpoint_name}")

        # Create the endpoint
        endpoint = Endpoint.create(
            display_name=endpoint_name,
            credentials=global_config.credentials,
            dedicated_endpoint_enabled=config.DEDICATED_ENDPOINT_ENABLED
        )

        logger.info(f"Successfully created endpoint: {endpoint.resource_name}")

        # Save the endpoint resource name to a YAML file
        save_endpoint_resource_name(endpoint.resource_name)

        return endpoint

    except Exception as e:
        logger.error(f"Failed to create endpoint: {e}")
        raise


def save_endpoint_resource_name(resource_name: str) -> None:
    """
    Saves the endpoint resource name to a YAML file.

    Args:
        resource_name (str): The resource name of the created endpoint.
    """
    try:
        file_path = "./config/endpoints.yml"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Read existing data from the YAML file if it exists
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = yaml.safe_load(file) or {}
        else:
            data = {}

        # Add the new endpoint resource name
        data["endpoints"] = data.get("endpoints", [])
        data["endpoints"].append(resource_name)

        # Write back to the YAML file
        with open(file_path, "w") as file:
            yaml.safe_dump(data, file)

        logger.info(f"Endpoint resource name saved to {file_path}")

    except Exception as e:
        logger.error(f"Failed to save endpoint resource name to YAML: {e}")
        raise


def create_model() -> Model:
    """
    Uploads a model to Vertex AI Model Registry using configuration values.

    Returns:
        Model: The uploaded model object.
    """
    try:
        # Generate a unique model name with a timestamp
        model_name_with_time = f"{config.MODEL_DISPLAY_NAME}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        logger.info(f"Uploading model with display name: {model_name_with_time}")

        # Upload the model
        model = Model.upload(
            display_name=model_name_with_time,
            artifact_uri=f"gs://{config.BUCKET_NAME}/timesfm",
            serving_container_image_uri=config.SERVE_DOCKER_URI,
            serving_container_ports=[8080],
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_environment_variables={
                "MODEL_ID": f"google/{config.MODEL_NAME}",
                "DEPLOY_SOURCE": config.DEPLOY_SOURCE,
                "TIMESFM_HORIZON": str(config.HORIZON),
                "TIMESFM_BACKEND": config.TIMESFM_BACKEND,
            },
            credentials=global_config.credentials,
        )

        logger.info(f"Model uploaded successfully: {model.resource_name}")
        return model

    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        raise


def deploy_model():
    """
    Deploys the model to the endpoint using configuration values.
    """
    try:
        logger.info("Starting the model deployment process.")

        # Create the model and endpoint
        model = create_model()
        endpoint = create_endpoint()

        # Deploy the model to the endpoint
        logger.info(f"Deploying model {model.display_name} to endpoint {endpoint.display_name}.")
        model.deploy(
            endpoint=endpoint,
            machine_type=config.MACHINE_TYPE,
            accelerator_type=config.ACCELERATOR_TYPE,
            accelerator_count=config.ACCELERATOR_COUNT,
            deploy_request_timeout=config.DEPLOY_REQUEST_TIMEOUT,
            service_account=config.SERVICE_ACCOUNT,
            enable_access_logging=True,
            min_replica_count=1,
            sync=True
        )

        logger.info(f"Model deployed successfully to endpoint: {endpoint.resource_name}")

    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        raise


if __name__ == "__main__":
    try:
        logger.info("Starting the model upload process.")
        deploy_model()
    except Exception as main_error:
        logger.error(f"An error occurred in the main execution: {main_error}")
        raise
