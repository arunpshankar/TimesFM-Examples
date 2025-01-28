from src.config.logging import logger
from src.config.setup import config
from urllib.parse import urlparse
from google.cloud import storage


def copy_model_artifacts(source_uri: str, destination_uri: str) -> None:
    """
    Copies model artifacts from a source URI to a destination URI using Google Cloud SDK.

    Args:
        source_uri (str): The source GCS URI (e.g., gs://bucket-name/path-to-source).
        destination_uri (str): The destination GCS URI (e.g., gs://bucket-name/path-to-destination).

    Raises:
        RuntimeError: If the copy operation fails.
    """
    try:
        logger.info("Starting to copy model artifacts from %s to %s", source_uri, destination_uri)

        # Parse URIs using urlparse
        source_parsed = urlparse(source_uri)
        destination_parsed = urlparse(destination_uri)

        if source_parsed.scheme != "gs" or destination_parsed.scheme != "gs":
            raise ValueError("Both source and destination URIs must start with 'gs://'")

        source_bucket_name = source_parsed.netloc
        source_prefix = source_parsed.path.lstrip("/")

        destination_bucket_name = destination_parsed.netloc
        destination_prefix = destination_parsed.path.lstrip("/")

        # Initialize the Google Cloud Storage client
        client = storage.Client()
        source_bucket = client.bucket(source_bucket_name)
        destination_bucket = client.bucket(destination_bucket_name)

        # List blobs in the source bucket with the specified prefix
        blobs = client.list_blobs(source_bucket, prefix=source_prefix)

        for blob in blobs:
            source_blob_name = blob.name
            destination_blob_name = source_blob_name.replace(source_prefix, destination_prefix, 1)

            # Copy the blob
            source_blob = source_bucket.blob(source_blob_name)
            destination_blob = destination_bucket.blob(destination_blob_name)

            # Rewrite the blob to the destination
            destination_blob.rewrite(source_blob)

        logger.info("Model artifacts copied successfully from %s to %s", source_uri, destination_uri)

    except Exception as e:
        logger.error("Failed to copy model artifacts: %s", str(e))
        raise RuntimeError(f"Failed to copy model artifacts from {source_uri} to {destination_uri}: {e}") from e


if __name__ == "__main__":
    source = f"{config.MODEL_LOCATION}/{config.MODEL_NAME}"
    destination = f"gs://{config.BUCKET_NAME}/timesfm"

    try:
        copy_model_artifacts(source, destination)
    except RuntimeError as e:
        logger.error("Error during artifact copy: %s", str(e))
