"""
Configuration module for house price prediction app.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Model parameters
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2
}

# Data generation parameters
DATA_CONFIG = {
    "n_samples": 1000,
    "random_seed": 50
}

# App configuration
APP_CONFIG = {
    "title": "House Price Prediction",
    "description": "Predict house prices based on features",
    "size_range": (1000, 3500),
    "bedrooms_range": (1, 5),
    "bathrooms_range": (1, 3),
    "default_size": 2000,
    "default_bedrooms": 2,
    "default_bathrooms": 2
} 