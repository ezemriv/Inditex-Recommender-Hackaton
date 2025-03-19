# Script to cast down train and test data

from config import PROCESSED_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH
import polars as pl
import os

def caster(df, train=True):
    print(f"Initial Size: {df.estimated_size('mb')} MB")
    
    df = df.with_columns([
        pl.col("session_id").cast(pl.UInt32),
        pl.col("user_id").cast(pl.UInt32),
        pl.col("country").cast(pl.UInt8),
        pl.col("partnumber").cast(pl.UInt16),
        pl.col("device_type").cast(pl.UInt8),
        pl.col("pagetype").cast(pl.UInt8),
    ])

    if train:
        df = df.with_columns([
            pl.col("add_to_cart").cast(pl.UInt8)
        ])

    print(f"Final Size: {df.estimated_size('mb')} MB")
    return df  # Return the modified DataFrame

def main():
    print("Processing train and test data...")

    # Train
    train_df = pl.read_csv(TRAIN_DATA_PATH, low_memory=True,
                           batch_size=100_000, 
                           try_parse_dates=True,
                          )
    train_df = caster(train_df, train=True)  # Assign the modified DataFrame

    # Test
    test_df = pl.read_csv(TEST_DATA_PATH, low_memory=True,
                          try_parse_dates=True,
                         )
    test_df = caster(test_df, train=False)  # Assign the modified DataFrame

    # Save to parquet
    train_df.write_parquet(os.path.join(PROCESSED_PATH, "train.parquet"), compression="zstd")
    test_df.write_parquet(os.path.join(PROCESSED_PATH, "test.parquet"), compression="zstd")

    print("Train and test data processed.")

if __name__ == "__main__":
    main()
