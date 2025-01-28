import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def load_forecast_json(file_path):
    """
    Loads JSON data and converts it into a DataFrame.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        pd.DataFrame: Processed DataFrame with forecast data.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Flatten nested JSON if necessary
        rows = []
        for entry in data[0]:
            for timestamp, forecast in zip(entry['timestamp'], entry['point_forecast']):
                rows.append({'timestamp': timestamp, 'forecast': forecast})

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return pd.DataFrame()

def visualize_forecasts_from_json(file_with_cov, file_without_cov, output_dir='./data/visuals'):
    """
    Reads forecast data from JSON files, creates visualizations, and saves them to a specified directory.

    Args:
        file_with_cov (str): Path to the JSON file containing forecasts with covariates.
        file_without_cov (str): Path to the JSON file containing forecasts without covariates.
        output_dir (str): Directory to save the visualizations.
    """
    # Load the data
    df_with_cov = load_forecast_json(file_with_cov)
    df_without_cov = load_forecast_json(file_without_cov)

    # Ensure timestamp is in datetime format and set as index
    if not df_with_cov.empty:
        df_with_cov['timestamp'] = pd.to_datetime(df_with_cov['timestamp'])
        df_with_cov.set_index('timestamp', inplace=True)

    if not df_without_cov.empty:
        df_without_cov['timestamp'] = pd.to_datetime(df_without_cov['timestamp'])
        df_without_cov.set_index('timestamp', inplace=True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot for "With Covariates"
    if not df_with_cov.empty:
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=df_with_cov, x=df_with_cov.index, y='forecast', label="With Covariates", marker="o", color="#4c72b0", linewidth=2)
        plt.title("Forecast: With Covariates", fontsize=18, color="#333333")
        plt.xlabel("Timestamp", fontsize=14, labelpad=10)
        plt.ylabel("Forecast Value", fontsize=14, labelpad=10)
        plt.grid(True, linestyle='--', linewidth=1.5, alpha=0.8)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'forecast_with_covariates.png'))

    # Plot for "Without Covariates"
    if not df_without_cov.empty:
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=df_without_cov, x=df_without_cov.index, y='forecast', label="Without Covariates", marker="o", color="#dd8452", linewidth=2)
        plt.title("Forecast: Without Covariates", fontsize=18, color="#333333")
        plt.xlabel("Timestamp", fontsize=14, labelpad=10)
        plt.ylabel("Forecast Value", fontsize=14, labelpad=10)
        plt.grid(True, linestyle='--', linewidth=1.5, alpha=0.8)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'forecast_without_covariates.png'))

# Example usage
# Replace the file paths with the actual paths to your JSON files
visualize_forecasts_from_json('./data/output/forecasts/raw_forecast_batch_1.json', './data/output/forecasts/cov_forecast_batch_1.json', output_dir='./data/visuals')
