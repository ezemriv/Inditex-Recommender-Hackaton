from config import (PRODUCTS_PARQUET_PATH_IMPUTED, 
                    PROD_SIM_MATRIX_PATH, 
                    PRODUCTS_ENGINEERED_PATH,
                    TRAIN_PARQUET_PATH, TEST_PARQUET_PATH,
                    TRAIN_ENGINEERED_PATH, TEST_ENGINEERED_PATH,
                    USERS_DATA_PATH, USERS_ENGINEERED_PATH)

from src.models.prepare_data import PRODUCTS, TRAIN_TEST, USERS
import polars as pl
from polars import StringCache

# Configurable options
PROCESS_PRODS = True
PROCESS_TRAIN = False 
SAMPLE_TRAIN = False
PROCESS_USERS = True
USE_CACHED_CANDIDATES = False  # Use precomputed candidates for test data
TOP_N_GLOBAL = 50  # Number of global most popular candidates

def main():
    if PROCESS_PRODS:
        print("Processing product data...")
        products = PRODUCTS(PRODUCTS_PARQUET_PATH_IMPUTED, 
                            PROD_SIM_MATRIX_PATH, 
                            PRODUCTS_ENGINEERED_PATH)
        products_df = products.run()

    if PROCESS_USERS:
        print("Processing user data...")
        users = USERS(USERS_DATA_PATH, USERS_ENGINEERED_PATH)
        users_df = users.run()

    with StringCache():
        # Initialize processor
        traintest_processor = TRAIN_TEST(
            sampling=SAMPLE_TRAIN, 
            train_path=TRAIN_PARQUET_PATH, 
            test_path=TEST_PARQUET_PATH,
            save_train_path=TRAIN_ENGINEERED_PATH,
            save_test_path=TEST_ENGINEERED_PATH,
            topN_global=TOP_N_GLOBAL
        )

        if USE_CACHED_CANDIDATES:
            print("Loading precomputed candidates...")
            precomputed_candidates = traintest_processor.select_global_candidates(
                pl.read_parquet(TRAIN_ENGINEERED_PATH)
            )
        else:
            precomputed_candidates = None

        # Run processing
        train, test = traintest_processor.run(
            process_train=PROCESS_TRAIN, 
            candidates_from_train=precomputed_candidates
        )

if __name__ == "__main__":
    main()
