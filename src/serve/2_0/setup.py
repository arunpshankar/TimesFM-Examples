from huggingface_hub import snapshot_download
from src.config.logging import logger
from src.config.setup import config
from google.cloud import storage
from typing import Optional
from typing import Dict 
from tqdm import tqdm
import time
import yaml
import os


class ModelManager:
    """
    A utility class to manage the downloading and uploading of models and LoRA adapters
    for deployment on Google Cloud Vertex AI.

    Attributes:
    -----------
    bucket_name : str
        The name of the GCS bucket where models will be uploaded.
    project_id : str
        The GCP project ID for the uploads.
    region : str
        The GCP region.

    Methods:
    --------
    download_model(repo_id: str, local_dir: str) -> None:
        Downloads a model or adapter from Hugging Face Hub to a local directory.
    upload_to_gcs(local_dir: str, gcs_path: Optional[str] = None) -> None:
        Uploads a local directory to a specified path in Google Cloud Storage.
    process_models(models: Dict[str, Dict[str, str]]) -> None:
        Processes a dictionary of models by downloading and uploading them.
    """

    def __init__(self):
        self.bucket_name = config.BUCKET_NAME
        self.client = storage.Client()
        self.project_id = config.PROJECT_ID
        self.region = config.REGION
        self.hf_token = self._load_hf_token()
        self._ensure_bucket_exists()

    def _load_hf_token(self) -> str:
        """
        Loads the Hugging Face token from the credentials file.

        Returns:
        --------
            str: The Hugging Face token.

        Raises:
        -------
            FileNotFoundError: If the credentials file does not exist.
            KeyError: If the `hf_token` key is missing in the YAML file.
        """
        credentials_path = os.path.join("credentials", "hf.yml")
        if not os.path.exists(credentials_path):
            logger.error("Hugging Face credentials file not found.")
            raise FileNotFoundError(f"Credentials file {credentials_path} does not exist.")

        try:
            with open(credentials_path, "r") as file:
                credentials = yaml.safe_load(file)
            hf_token = credentials["hf_token"]
            logger.info("Successfully loaded Hugging Face token.")
            return hf_token
        except KeyError:
            logger.error("Hugging Face token not found in the credentials file.")
            raise KeyError("Hugging Face token not found in the credentials file.")
        except Exception as e:
            logger.error(f"Error reading Hugging Face credentials: {str(e)}")
            raise Exception(f"Error reading Hugging Face credentials: {str(e)}") from e

    def _ensure_bucket_exists(self) -> None:
        """
        Ensures the GCS bucket exists. If not, creates the bucket.

        Raises:
        -------
            Exception: For any errors during bucket creation.
        """
        try:
            bucket = self.client.lookup_bucket(self.bucket_name)
            if bucket is None:
                logger.info(f"Bucket {self.bucket_name} does not exist. Creating bucket...")
                bucket = self.client.create_bucket(self.bucket_name, location=self.region)
                logger.info(f"Bucket {self.bucket_name} created successfully in region {self.region}.")
            else:
                logger.info(f"Bucket {self.bucket_name} already exists.")
        except Exception as e:
            logger.error(f"Failed to ensure bucket existence: {str(e)}")
            raise Exception(f"Failed to ensure bucket existence: {str(e)}") from e

    def download_model(self, repo_id: str, local_dir: str) -> None:
        """
        Downloads a model or adapter from Hugging Face Hub to a local directory.

        Args:
        -----
            repo_id : str
                The Hugging Face repository ID.
            local_dir : str
                The local directory where the files will be stored.

        Raises:
        -------
            Exception: For any errors during the download process.
        """
        try:
            logger.info(f"Starting download of {repo_id} to {local_dir}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                token=self.hf_token,  # Use the token for authentication
                ignore_patterns=["*.lock"]
            )
            logger.info(f"Successfully downloaded {repo_id} to {local_dir}.")
        except Exception as e:
            logger.error(f"Failed to download {repo_id}: {str(e)}")
            raise Exception(f"Failed to download {repo_id}: {str(e)}") from e

    def upload_to_gcs(self, local_dir: str, gcs_path: Optional[str] = None) -> None:
        """
        Uploads a local directory to a specified path in Google Cloud Storage.

        Args:
        -----
            local_dir : str
                The local directory to upload.
            gcs_path : Optional[str]
                The GCS path to upload to. Defaults to the root of the bucket.

        Raises:
        -------
            Exception: For any errors during the upload process.
        """
        if not os.path.exists(local_dir):
            logger.error(f"Local directory {local_dir} does not exist.")
            raise FileNotFoundError(f"Local directory {local_dir} does not exist.")

        gcs_path = gcs_path or ""
        bucket = self.client.bucket(self.bucket_name)

        try:
            # Collect all files to upload
            files_to_upload = []
            for root, _, files in os.walk(local_dir):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, local_dir)
                    blob_path = os.path.join(gcs_path, relative_path)
                    files_to_upload.append((local_file_path, blob_path))

            # Use tqdm for progress tracking
            with tqdm(total=len(files_to_upload), desc="Uploading files to GCS", unit="file") as progress_bar:
                for local_file_path, blob_path in files_to_upload:
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(local_file_path)

                    # Log the upload and update the progress bar
                    logger.info(f"Uploaded {local_file_path} to gs://{self.bucket_name}/{blob_path}")
                    progress_bar.update(1)
        except Exception as e:
            logger.error(f"Failed to upload files from {local_dir} to GCS: {str(e)}")
            raise Exception(f"Failed to upload files from {local_dir} to GCS: {str(e)}") from e

    def process_models(self, models: Dict[str, Dict[str, str]]) -> None:
        """
        Iterates through the models dictionary, downloading from Hugging Face Hub
        and uploading to Google Cloud Storage.

        Args:
        -----
            models : Dict[str, Dict[str, str]]
                A dictionary where keys are model names and values are dictionaries containing
                `repo_id` and `local_dir`.

        Raises:
        -------
            Exception: If any error occurs during processing, logs the error but continues processing.
        """
        for model_name, model_info in models.items():
            logger.info(f"Processing model: {model_name}")
            start_time = time.time()
            try:
                # Validate model_info structure
                if not isinstance(model_info, Dict) or 'repo_id' not in model_info or 'local_dir' not in model_info:
                    logger.error(f"Invalid model info for {model_name}: {model_info}")
                    continue

                # Extract necessary information
                repo_id: str = model_info['repo_id']
                local_dir: str = model_info['local_dir']

                # Download the model or adapter
                download_start = time.time()
                self.download_model(repo_id, local_dir)
                download_end = time.time()
                logger.info(
                    f"Model {model_name} downloaded successfully. Time taken: {download_end - download_start:.2f} seconds."
                )

                # Upload the model or adapter to GCS
                upload_start = time.time()
                self.upload_to_gcs(local_dir, model_name)
                upload_end = time.time()
                logger.info(
                    f"Model {model_name} uploaded successfully. Time taken: {upload_end - upload_start:.2f} seconds."
                )

                total_time = time.time() - start_time
                logger.info(f"Successfully processed model: {model_name} in {total_time:.2f} seconds.")
            except Exception as e:
                logger.error(f"Failed to process model {model_name}: {str(e)}")
                continue


if __name__ == "__main__":
    manager = ModelManager()
    models = {"timesfm_2_0": {"repo_id": "google/timesfm-2.0-500m-pytorch", "local_dir": "./model"}}
    manager.process_models(models=models)