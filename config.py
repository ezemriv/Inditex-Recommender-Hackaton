import os

# Project Root
ROOT = os.path.dirname(os.path.abspath(__file__))

# Data Paths
DATA_PATH = os.path.join(ROOT, 'data')
RAW_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')

# Raw Data Files
PRODUCTS_DATA_PATH = os.path.join(RAW_PATH, 'products.pkl')
TRAIN_DATA_PATH = os.path.join(RAW_PATH, 'train.csv')
TEST_DATA_PATH = os.path.join(RAW_PATH, 'test.csv')

# Processed Data Files
# USERS_DATA_PATH = os.path.join(PROCESSED_PATH, 'users.parquet')
USERS_DATA_PATH = os.path.join(PROCESSED_PATH, 'users_combined.parquet')
TRAIN_PARQUET_PATH = os.path.join(PROCESSED_PATH, 'train.parquet')
TEST_PARQUET_PATH = os.path.join(PROCESSED_PATH, 'test.parquet')
PRODUCTS_PARQUET_PATH = os.path.join(PROCESSED_PATH, 'products.parquet')

# Model Paths
MODELS_PATH = os.path.join(ROOT, 'models')

# Prediction Paths
PREDICTIONS_PATH = os.path.join(ROOT, 'predictions')
EX_SUBMISSION_1_PATH = os.path.join(PREDICTIONS_PATH, 'example_predictions_1.json')
EX_SUBMISSION_3_PATH = os.path.join(PREDICTIONS_PATH, 'example_predictions_3.json')
SUBMISSION_1_PATH = os.path.join(PREDICTIONS_PATH, 'predictions_1.json')
SUBMISSION_3_PATH = os.path.join(PREDICTIONS_PATH, 'predictions_3.json')

# API Configuration
BASE_API_URL = "https://zara-boost-hackathon.nuwe.io"
GET_ALL_USERS_ENDPOINT = f"{BASE_API_URL}/users"