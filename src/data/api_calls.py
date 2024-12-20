import requests
import time

from config import GET_ALL_USERS_ENDPOINT

def fetch_all_user_ids(retries=5, delay=2):
    # url = 'https://zara-boost-hackathon.nuwe.io/users'
    url = GET_ALL_USERS_ENDPOINT
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()  # Assuming API returns a list of user IDs
            print(f"Attempt {attempt + 1}: Failed with status {response.status_code}. Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}: Error occurred - {e}. Retrying...")
        time.sleep(delay)
    
    print("Failed to fetch all user IDs after multiple attempts.")
    return None

def fetch_user_data(user_id, retries=5, delay=2):
    url = GET_ALL_USERS_ENDPOINT+f'/{user_id}'
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'values' in data and 'R' in data['values']:
                    idx = data['values']['R'].index(min(data['values']['R']))
                    data['values'] = {k: [v[idx]] for k, v in data['values'].items()}
                return data
            print(f"Attempt {attempt + 1}: Failed with status {response.status_code}. Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}: Error occurred - {e}. Retrying...")
        time.sleep(delay)
    
    print(f"Failed to fetch data for user_id {user_id} after {retries} attempts.")
    return None
