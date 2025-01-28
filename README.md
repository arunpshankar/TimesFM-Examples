# TimesFM Examples

A collection of examples showcasing **TimesFM**'s time series forecasting capabilities using **TimesFM 2.0**. This repository includes Python modules and Jupyter notebooks for:

1. **Serving**  
2. **Inference**  
3. **Fine-Tuning**  

[time](data/visuals/time.jpeg)

#

## Table of Contents

- [TimesFM Examples](#timesfm-examples)
- [](#)
  - [Table of Contents](#table-of-contents)
- [](#-1)
  - [Overview](#overview)
- [](#-2)
  - [Features](#features)
- [](#-3)
  - [Prerequisites](#prerequisites)
- [](#-4)
  - [Installation](#installation)
- [](#-5)
  - [Repository Structure](#repository-structure)
- [](#-6)
  - [Model Serving](#model-serving)
- [](#-7)
  - [Inference](#inference)
- [](#-8)
  - [Fine-Tuning](#fine-tuning)

#

## Overview

**TimesFM Playground** provides a hands-on environment to explore and experiment with time series forecasting using TimesFM. You can deploy a pre-trained model to Google Vertex AI for real-time inference or fine-tune a model for specific datasets such as stock prices, temperatures, or other custom time series.

#

## Features

- **Model Serving**:  
  - End-to-end workflow to deploy a TimesFM model on **Vertex AI**.
  - Real-time synchronous inference via a deployed endpoint.

- **Inference**:  
  - Demonstrations of forecasting various scenarios (e.g., stock prices, temperature).
  - Handling multivariate forecasting with covariates.
  - Basic anomaly detection on output predictions.
  - Visualization scripts for better interpretability.

- **Fine-Tuning**:  
  - Notebook-based pipeline for preparing custom datasets and fine-tuning **TimesFM 2.0**.
  - Example includes a stock price dataset (using `yfinance`) with improvements in prediction accuracy.

#

## Prerequisites

1. **Google Cloud Platform**  
   - A GCP project with the following services enabled:  
     - Vertex AI  
     - Cloud Storage  
   - A service account with the required permissions to read/write models, create endpoints, etc.

2. **Python 3.7+**  
   - Make sure you have Python 3.7 or above installed.

3. **Git**  
   - To clone the repository.

4. **Credentials Folder**  
   - A local folder named `credentials` containing your GCP service account key (JSON file).

#

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/arunpshankar/TimesFM-Playground.git
   cd TimesFM-Playground
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .timesfm
   source .timesfm/bin/activate
   ```

3. **Upgrade pip and install required packages**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set environment variables** (to avoid writing `.pyc` files and ensure `src` is on `PYTHONPATH`):
   ```bash
   export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PYTHONPATH:.
   ```

5. **GCP Setup**:
   - Create or choose a **Google Cloud Storage** bucket to store model artifacts.
   - Ensure your service account key JSON file is placed under the `credentials` folder.
   - Confirm your service account has the necessary permissions within your GCP project.

#

## Repository Structure

```
TimesFM-Playground/
├── config/
│   ├── serving_params.yml          # Configuration for serving
│   └── ...
├── credentials/                    # GCP service account key goes here
├── data/
│   ├── input/                      # Input datasets for inference
│   ├── output/                     # Forecast output
│   └── visuals/                    # Visualizations and plots
├── notebooks/
│   └── finetune_timesfm.ipynb      # Jupyter notebook for fine-tuning
├── src/
│   ├── serve/
│   │   ├── setup.py                # Copies model artifacts to GCS
│   │   └── deploy.py               # Deploys the model on Vertex AI
│   ├── invoke/
│   │   └── ...                     # Inference examples and scripts
│   └── ...
└── requirements.txt
```

#

## Model Serving

1. **Pre-Setup**  
   - Make sure you have a GCS bucket created (via the UI or CLI).
   - Have your service account JSON in `credentials/`.

2. **Deploy Process**  
   1. **Setup model artifacts**:
      ```bash
      python src/serve/setup.py
      ```
      This script fetches the TimesFM model artifacts from a public cloud storage URI and copies them to your chosen GCS bucket.

   2. **Deploy to Vertex AI**:
      ```bash
      python src/serve/deploy.py
      ```
      This script:
      - Registers the model in **Vertex AI** Model Registry.
      - Creates an **endpoint**.
      - Deploys the model to the endpoint, making it ready to receive real-time inference calls.

3. **Inference Endpoint**  
   - Once deployed, you will have a Vertex AI endpoint URL.
   - This endpoint can be used to make **POST** requests for time series forecasts.

#

## Inference

Inside `src/invoke`, you will find multiple Python scripts demonstrating how to:

- **Invoke the Vertex AI endpoint** with different types of payloads (univariate or multivariate).
- **Predict** various scenarios, such as:
  - Stock prices
  - Weekly temperatures
  - Custom time series
- **Covariates**: Learn how to pass additional features to improve forecasting.
- **Anomaly Detection**: Identify anomalies in the forecasted data.
- **Visualization**: The scripts also include code to generate plots for intuitive understanding of forecast results.

#

## Fine-Tuning

A single Jupyter notebook (`finetune_timesfm.ipynb`) encapsulates all the steps needed to:

1. **Prepare a dataset** (e.g., stock prices from `yfinance`).  
2. **Fine-tune TimesFM 2.0** on the dataset.  
3. **Evaluate** the performance of the fine-tuned model.

> **Note**: You can run this notebook on **Vertex AI Workbench**, **Google Colab**, or any environment with proper GPU/TPU resources.