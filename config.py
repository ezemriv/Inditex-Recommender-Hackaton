import os

# Project Root
ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_PATH = os.path.join(ROOT, 'data')
RAW_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')
MODELS_PATH = os.path.join(ROOT, 'models')
PREDICTIONS_PATH = os.path.join(ROOT, 'predictions')

# File Names
PRODUCTS_DATA_PATH = os.path.join(RAW_PATH, 'products.pkl')
USERS_DATA_PATH = os.path.join(PROCESSED_PATH, 'users.parquet')
TRAIN_DATA_PATH = os.path.join(RAW_PATH, 'train.csv')
TEST_DATA_PATH = os.path.join(RAW_PATH, 'test.csv')
TRAIN_PARQUET_PATH = os.path.join(PROCESSED_PATH, 'train.parquet')
TEST_PARQUET_PATH = os.path.join(PROCESSED_PATH, 'test.parquet')
SUBMISSION_1_PATH = os.path.join(PREDICTIONS_PATH, 'example_predictions_1.json')
SUBMISSION_3_PATH = os.path.join(PREDICTIONS_PATH, 'example_predictions_3.json')
PRODUCTS_PARQUET_PATH = os.path.join(PROCESSED_PATH, 'products.parquet')

# API Configuration
BASE_API_URL = "https://zara-boost-hackathon.nuwe.io"
GET_ALL_USERS_ENDPOINT = f"{BASE_API_URL}/users"