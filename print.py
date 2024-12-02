import requests
import pandas as pd
from faker import Faker
import random
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

fake = Faker()

try:
    # Fetch data from APIs
    users_api = requests.get('http://127.0.0.1:8000/users/')
    users_api.raise_for_status()
    users_api_data = users_api.json()

    employees_api = requests.get('http://127.0.0.1:8000/employees/')
    employees_api.raise_for_status()
    employees_api_data = employees_api.json()

    # Convert API responses to DataFrames
    users_df = pd.DataFrame(users_api_data if isinstance(users_api_data, list) else users_api_data.get('users', []))
    employees_df = pd.DataFrame(employees_api_data if isinstance(employees_api_data, list) else employees_api_data.get('employees', []))

    # Merge DataFrames on 'user_id'
    merged_df = pd.merge(employees_df, users_df, on='user_id', how='inner')

    # Describe the dataset
    print("Dataset Description:")
    print(f"Columns: {merged_df.columns.tolist()}")
    print(f"Data Types: {merged_df.dtypes}")
    print(f"Missing Values:\n{merged_df.isnull().sum()}")
    print(f"Summary Statistics:\n{merged_df.describe(include='all')}")
    print(f"Shape: {merged_df.shape}")

    # Handle missing values
    merged_df.fillna({
        'position': 'Unknown',
        'hire_date': '1900-01-01',  # Default for missing dates
        'phone_number': 'Unknown',
        'emergency_contact': 'Unknown',
        'email_address': 'Unknown'
    }, inplace=True)

    # Convert hire_date to datetime
    merged_df['hire_date'] = pd.to_datetime(merged_df['hire_date'], errors='coerce')

    # Ensure the dataset contains 500,000 rows
    current_rows = merged_df.shape[0]
    if current_rows <= 500000:
        num_synthetic_records = 500000 - current_rows
        print(f"Generating {num_synthetic_records} synthetic records...")
        synthetic_data = [
            {
                'id': fake.random_int(min=100000, max=999999),
                'name': fake.name(),
                'position': random.choice(['Manager', 'Engineer', 'Analyst', 'Clerk']),
                'hire_date': fake.date_between(start_date='-10y', end_date='today'),
                'phone_number': fake.phone_number(),
                'emergency_contact': fake.phone_number(),
                'email_address': fake.email(),
                'user_id': fake.random_int(min=1, max=100000),
                'username': fake.user_name(),
                'email': fake.email()
            }
            for _ in range(num_synthetic_records)
        ]
        synthetic_df = pd.DataFrame(synthetic_data)
        merged_df = pd.concat([merged_df, synthetic_df], ignore_index=True)

    # Basic preprocessing: Normalize numeric fields
    # Add a sample numeric field for demonstration purposes
    merged_df['sample_numeric_field'] = random.choices(range(1000, 10000), k=len(merged_df))
    scaler = MinMaxScaler()
    merged_df[['sample_numeric_field']] = scaler.fit_transform(merged_df[['sample_numeric_field']])

    # Feature engineering
    # Convert hire_date to datetime if it's not already
    merged_df['hire_date'] = pd.to_datetime(merged_df['hire_date'], errors='coerce')
    
    # Calculate employment duration
    current_date = pd.Timestamp.now()
    merged_df['employment_duration_years'] = (current_date - merged_df['hire_date']).dt.days / 365

    merged_df['contact_availability'] = merged_df['phone_number'].apply(lambda x: 'Valid' if x != 'Unknown' else 'Missing')

    # Save the processed dataset
    merged_df.to_csv("processed_employees_users.csv", index=False)
    print("Processed dataset saved as 'processed_employees_users.csv'.")
    print(f"Final Dataset Shape: {merged_df.shape}")

except requests.exceptions.RequestException as e:
    print(f"Error fetching data from API: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

