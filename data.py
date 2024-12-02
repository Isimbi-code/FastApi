import requests
import pandas as pd
from faker import Faker
import random

fake = Faker()

def fetch_data():
    try:
        events_api = requests.get('http://127.0.0.1:8000/api/events')  
        events_api.raise_for_status()  
        events_api_data = events_api.json() 

        volunteers_api = requests.get('http://127.0.0.1:8000/api/volunteers')  
        volunteers_api.raise_for_status() 
        volunteers_api_data = volunteers_api.json()  

        return events_api_data, volunteers_api_data  

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return [], []  

def generate_synthetic_data(existing_records, total_records=500000):
    num_synthetic_records = total_records - len(existing_records)  # Calculate how many synthetic records to generate
    print(f"Generating {num_synthetic_records} synthetic records...")

    # Generate synthetic event data with random values
    synthetic_events = [
        {
            "id": i,
            "event_name": fake.catch_phrase(),
            "event_date": fake.date_between(start_date='-2y', end_date='today'),
            "event_location": fake.city(),
            "event_category": random.choice(["Education", "Health", "Environment", "Community"]),
            "numerical_column1": random.uniform(10, 100),
            "numerical_column2": random.uniform(20, 200),
            "event_end_date": fake.date_between(start_date='today', end_date='+1y') 
        }
        for i in range(1, num_synthetic_records + 1)
    ]

    synthetic_volunteers = [
        {
            "id": i,
            "volunteer_name": fake.name(),
            "volunteer_email": fake.email(),
            "hours_contributed": random.uniform(5, 50),
        }
        for i in range(1, num_synthetic_records + 1)
    ]

    return synthetic_events, synthetic_volunteers  

def merge_and_prepare_data(events_data, volunteers_data):
    edf = pd.DataFrame(events_data) 
    vdf = pd.DataFrame(volunteers_data)  

    merged_df = pd.merge(edf, vdf, on="id", how="inner")  # data pre-processing 

    merged_df.ffill(inplace=True) 

    merged_df.fillna({
        'event_name': 'Unknown', 
        'event_date': 'Unknown', 
        'event_location': 'Unknown', 
        'event_category': 'Unknown', 
        'numerical_column1': 0, 
        'numerical_column2': 0,
        'volunteer_name': 'Unknown',
        'volunteer_email': 'Unknown',
        'hours_contributed': 2,
        'event_end_date': 'Unknown'
    }, inplace=True)

    merged_df['event_date'] = pd.to_datetime(merged_df['event_date'], errors='coerce')
    merged_df['event_end_date'] = pd.to_datetime(merged_df['event_end_date'], errors='coerce')

    merged_df['event_duration'] = (merged_df['event_end_date'] - merged_df['event_date']).dt.days

    merged_df['efficiency_ratio'] = merged_df['hours_contributed'] / (merged_df['numerical_column1'] + merged_df['numerical_column2'])

    merged_df['volunteer_activity_level'] = pd.cut(merged_df['hours_contributed'],
                                                    bins=[0, 10, 25, 50],
                                                    labels=['Low', 'Medium', 'High'])

    merged_df['event_popularity_index'] = merged_df['numerical_column1'] + merged_df['numerical_column2']

  
    merged_df['volunteer_event_match'] = merged_df.apply(
        lambda row: 1 if (row['event_category'] == 'Education' and row['hours_contributed'] > 20) else 0, axis=1)

   
    merged_df['numerical_column1'] = merged_df['numerical_column1'].astype(float)
    merged_df['numerical_column2'] = merged_df['numerical_column2'].astype(float)
    merged_df['hours_contributed'] = merged_df['hours_contributed'].astype(float)

    merged_df = pd.get_dummies(merged_df, columns=['event_category', 'event_location'], drop_first=True)


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    merged_df[['numerical_column1', 'numerical_column2', 'hours_contributed']] = scaler.fit_transform(
        merged_df[['numerical_column1', 'numerical_column2', 'hours_contributed']]
    )

    return merged_df  

def describe_dataset(df):
    description = {
        "columns": df.columns.tolist(), 
        "dtypes": df.dtypes.to_dict(), 
        "summary_statistics": df.describe(),
        "missing_values": df.isnull().sum(),  
        "shape": df.shape  
    }
    return description

def main():
    output_file = "events_dataset.csv" 
    merged_data = pd.read_csv(output_file)  

    
    print(f"Shape of the dataset: {merged_data.shape}")
    
  
    print("\nSample of the merged data:")
    print(merged_data.head())

   
    dataset_description = describe_dataset(merged_data)
    
    print("\nDataset Description:")
    print(f"Columns: {dataset_description['columns']}")
    print(f"Data Types: {dataset_description['dtypes']}")
    print(f"Summary Statistics: {dataset_description['summary_statistics']}")
    print(f"Missing Values: {dataset_description['missing_values']}")
    print(f"Shape: {dataset_description['shape']}")

    return merged_data

if __name__ == "__main__":
    merged_data = main()  



# import csv
# import random
# from faker import Faker
# from datetime import datetime
# fake = Faker()
# num_records = 450000
# output_file = "events_dataset.csv"
# data = [
#     {
#         "title": fake.catch_phrase(),
#         "name": fake.name(),
#         "date": fake.date_between(start_date='-2y', end_date='today'),
#         "location": fake.city(),
#         "description": fake.text(max_nb_chars=200),
#         "created_at": fake.date_time_this_year(),
#     }
#     for _ in range(num_records)
# ]
# with open(output_file, mode="w", newline="", encoding="utf-8") as file:
#     writer = csv.DictWriter(file, fieldnames=["title", "name", "date", "location", "description", "created_at"])
#     writer.writeheader()
#     writer.writerows(data)
# print(f"Dataset with {num_records} records saved to {output_file}.")

