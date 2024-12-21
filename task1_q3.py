import polars as pl
import argparse

from config import TRAIN_DATA_PATH
from src.utils import timer

# Function to process the train dataset for task1-q4
@timer
def process_large_dataset(train_path, sampling):
    """
    Processes a large dataset to calculate the average number of visits 
    before a product is added to the cart.

    Args:
        train_path (str): Path to the training dataset.
        sampling (bool, optional): If True, loads only a sample of the data. 
                                  Defaults to True.

    Returns:
        None
    """
    try:
        if sampling:
            N_ROWS = 1_000_000
        else:
            N_ROWS = None

        # Load data lazily
        train = pl.scan_csv(train_path, low_memory=True,
                                n_rows=N_ROWS,
                                try_parse_dates=True,
                                ).select(["session_id", "partnumber", "add_to_cart", "timestamp_local"])
        
        print("Loaded data:", train.select(pl.len()).collect().item())

        # Products that were added to the cart
        products_added = (
            train
            .filter(pl.col("add_to_cart") == 1)
            .select("partnumber")
            .unique()  # Stay in LazyFrame
        )

        # Keep only interactions for these products
        interactions_for_cart_products = (
            train
            .join(products_added, on="partnumber", how="inner")
            .sort(["session_id", "partnumber", "timestamp_local"])
        )

        # Add a cumulative flag for add_to_cart in each group
        grouped_data = (
            interactions_for_cart_products
            .group_by(["session_id", "partnumber"])
            .agg([
                pl.col("add_to_cart").cum_sum().alias("add_to_cart_cumsum"),
                pl.col("add_to_cart"),
                pl.col("timestamp_local"),
            ])
            .explode(["add_to_cart_cumsum", "add_to_cart", "timestamp_local"])
        )

        # Calculate pre-cart visits
        pre_cart_visits = (
            grouped_data
            .filter(pl.col("add_to_cart_cumsum") == 0)
            .group_by("partnumber")
            .agg(pl.col("session_id").count().alias("visit_count"))
        )

        # Collect and calculate the average visits before add_to_cart
        final_result = pre_cart_visits.collect(streaming=True)
        average_visits = round(final_result["visit_count"].mean(), 2)

        print("Average Visits Before Adding to Cart:", average_visits)
    except Exception as e:
        print("An error occurred while processing the dataset:", e)

# Main function to execute from the command line
def main():
    parser = argparse.ArgumentParser(description="Process a large dataset with Polars.")
    parser.add_argument("--sample", help="Whether to use a sample of the data.")
    args = parser.parse_args()

    process_large_dataset(TRAIN_DATA_PATH, sampling=args.sample)

if __name__ == "__main__":
    main()