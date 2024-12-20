import os

# Project Root
ROOT = os.getcwd()

# Paths
DATA_PATH = os.path.join(ROOT, 'data')
RAW_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')
MODELS_PATH = os.path.join(ROOT, 'models')
PREDICTIONS_PATH = os.path.join(ROOT, 'predictions')

# File Names
PRODUCTS_DATA_PATH = os.path.join(RAW_PATH, 'products.pkl')
USERS_DATA_PATH = os.path.join(RAW_PATH, 'users.parquet')
TRAIN_DATA_PATH = os.path.join(RAW_PATH, 'train.csv')
TEST_DATA_PATH = os.path.join(RAW_PATH, 'test.csv')

# API Configuration
BASE_API_URL = "https://zara-boost-hackathon.nuwe.io"
GET_ALL_USERS_ENDPOINT = f"{BASE_API_URL}/users"