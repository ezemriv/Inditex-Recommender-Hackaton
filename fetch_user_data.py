# Script to fetch all users data from users API using batches and save them as parquet files

import os
import pandas as pd
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import timer
from src.data.api_calls import fetch_all_user_ids, fetch_user_data
from config import RAW_PATH, PROCESSED_PATH

def chunk_list(data, chunk_size=100):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

@timer
def get_all_users_info_in_batches(chunk_size=100):
    user_ids = fetch_all_user_ids()
    if not user_ids:
        print("No user data fetched.")
        return

    batch_path = os.path.join(RAW_PATH, "user_batches")
    os.makedirs(batch_path, exist_ok=True)

    batch_number = 1
    for chunk in chunk_list(user_ids, chunk_size):
        print(f"Processing batch {batch_number}...")
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_user_data, uid): uid for uid in chunk}
            for future in as_completed(futures):
                user_data = future.result()
                if user_data and 'values' in user_data:
                    user_df = pd.DataFrame.from_dict(user_data['values'])
                    user_df['user_id'] = futures[future]
                    results.append(user_df)
        
        if results:
            batch_df = pd.concat(results, ignore_index=True)
            parquet_file = os.path.join(batch_path, f"user_batch_{batch_number}.parquet")
            batch_df.to_parquet(parquet_file, index=False)
            print(f"Saved batch {batch_number} to {parquet_file}")
        else:
            print(f"No data in batch {batch_number}")
        
        batch_number += 1

@timer
def merge_batches(batch_path):
    print("Merging batches...")
    all_users_df = pl.DataFrame()
    
    for file_name in os.listdir(batch_path):
        if file_name.endswith('.parquet'):
            file_path = os.path.join(batch_path, file_name)
            data = pl.read_parquet(file_path)
            all_users_df = pl.concat([all_users_df, data])
            print(f"Processed {file_name}")
    
    # Cast columns to smaller types
    all_users_df = all_users_df.with_columns([
        pl.col('country').cast(pl.Int8),
        pl.col('R').cast(pl.Int16),
        pl.col('F').cast(pl.Int16), 
        pl.col('M').cast(pl.Float32),
        pl.col('user_id').cast(pl.Int32)
    ])

    output_path = os.path.join(PROCESSED_PATH, "users_combined.parquet")
    all_users_df.write_parquet(output_path)
    print(f"Final combined DataFrame saved at {output_path}")
    return all_users_df

# Function to only fetch failed users

def fetch_failed_users(hardcoded_failed_ids, output_file):
    if not hardcoded_failed_ids:
        print("No failed user IDs provided.")
        return

    print(f"Fetching data for {len(hardcoded_failed_ids)} failed users...")

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_user_data, uid): uid for uid in hardcoded_failed_ids}
        for future in as_completed(futures):
            try:
                user_data = future.result()
                if user_data and 'values' in user_data:
                    user_df = pd.DataFrame.from_dict(user_data['values'])
                    user_df['user_id'] = futures[future]
                    results.append(user_df)
            except Exception as e:
                print(f"User {futures[future]} retry failed: {e}")
    
    if results:
        retry_df = pd.concat(results, ignore_index=True)
        print(retry_df)
        output_path = os.path.join(RAW_PATH, "user_batches", output_file)
        retry_df.to_parquet(output_path, index=False)
        print(f"Saved retried data to {output_path}")
    else:
        print("No data retrieved for failed users.")

def main():
    batch_path = os.path.join(RAW_PATH, "user_batches")
    # print("Fetching all user data in batches...")
    # print("Total number of users:", len(fetch_all_user_ids()))
    # get_all_users_info_in_batches(chunk_size=100_000)
    # print("All batches processed.")
    # merge_batches(batch_path)
    # print("All batches merged.")

    # Fetch failed users
    # Hardcoded list of failed user IDs
    hardcoded_failed_ids = [203225] 
    output_parquet_filename = "failed_users.parquet"
    fetch_failed_users(hardcoded_failed_ids, output_parquet_filename)
    merge_batches(batch_path)
    print("All batches merged.")

if __name__ == "__main__":
    main()
