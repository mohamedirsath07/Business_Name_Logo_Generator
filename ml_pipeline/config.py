# Configuration file for ML Pipeline
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model configurations
BUSINESS_NAME_MODEL_CONFIG = {
    "model_type": "transformer",
    "max_length": 50,
    "vocab_size": 10000,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "early_stopping_patience": 5
}

LOGO_GENERATION_CONFIG = {
    "model_type": "vae",
    "latent_dim": 128,
    "image_size": 256,
    "channels": 3,
    "learning_rate": 0.0002,
    "batch_size": 16,
    "epochs": 100,
    "beta": 1.0  # For VAE loss
}

# API configurations
API_CONFIG = {
    "host": "localhost",
    "port": 5000,
    "debug": True
}

# Data sources
DATA_SOURCES = {
    "business_names": [
        "https://raw.githubusercontent.com/datasets/company-names/master/data/companies.csv",
        # Add more data sources as needed
    ],
    "logos": [
        # We'll use generated synthetic data for logos
    ]
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "ml_pipeline.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}
