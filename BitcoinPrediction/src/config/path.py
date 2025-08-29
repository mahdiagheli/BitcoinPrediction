import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir , os.pardir))

RAW_DATA_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR         = os.path.join(BASE_DIR, "models")
SCATTER_DIR        = os.path.join(BASE_DIR, "outputs", "Scatter_out")
