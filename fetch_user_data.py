# Script to fetch all users data from users API using batches and save them as parquet files

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import timer
from src.data.api_calls import fetch_all_user_ids, fetch_user_data
from config import RAW_PATH

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

def main():
    print("Fetching all user data in batches...")
    print("Total number of users:", len(fetch_all_user_ids()))

    get_all_users_info_in_batches(chunk_size=100_000)
    
    print("All batches processed.")

if __name__ == "__main__":
    main()
