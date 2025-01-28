import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
from src.config.logging import logger 


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

def visualize_first_batch(df, batch_size=128, context_len=5, horizon_len=2, output_dir='./data/visuals'):
    """
    Processes the data into batches and visualizes the first batch of context and forecast data.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        batch_size (int): Number of examples per batch.
        context_len (int): Length of the context window.
        horizon_len (int): Length of the forecast horizon.
        output_dir (str): Directory to save the visualizations.
    """
    # Prepare the examples
    examples = defaultdict(list)
    num_examples = 0

    # Process the data as per the batching logic
    for start in range(0, len(df) - (context_len + horizon_len), horizon_len):
        num_examples += 1
        examples["inputs"].append(df["y"][start:(context_end := start + context_len)].tolist())
        examples["outputs"].append(df["y"][context_end:(context_end + horizon_len)].tolist())
        examples["timestamps"].append(df["ds"][start:context_end + horizon_len].tolist())

    # Extract the first batch
    first_batch = {
        "inputs": examples["inputs"][:batch_size],
        "outputs": examples["outputs"][:batch_size],
        "timestamps": examples["timestamps"][:batch_size],
    }

    # Flatten the first batch for visualization
    batch_df = pd.DataFrame({
        "timestamp": [ts for sublist in first_batch["timestamps"] for ts in sublist],
        "meantemp": [val for sublist in (first_batch["inputs"] + first_batch["outputs"]) for val in sublist],
        "type": ["context"] * (len(first_batch["inputs"][0]) * len(first_batch["inputs"])) +
                ["forecast"] * (len(first_batch["outputs"][0]) * len(first_batch["outputs"]))
    })

    # Convert timestamp to datetime for plotting
    batch_df["timestamp"] = pd.to_datetime(batch_df["timestamp"])

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Visualize the first batch
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=batch_df, x="timestamp", y="meantemp", hue="type", marker="o")
    plt.title("First Batch Visualization: Context and Forecast", fontsize=18)
    plt.xlabel("Timestamp", fontsize=14)
    plt.ylabel("Mean Temperature", fontsize=14)
    plt.grid(True, linestyle='--', linewidth=1.5, alpha=0.8)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Data Type")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, 'first_batch_visualization.png')
    plt.savefig(output_file)
    plt.show()
    print(f"Visualization saved to {output_file}")

# Example usage
# Assuming 'df' is already loaded with the required data
df = load_electricity_data('./data/input/electricity.csv')
visualize_first_batch(df, batch_size=128, context_len=5, horizon_len=2, output_dir='./data/visuals')