import psycopg2
import csv
from datetime import datetime
conn = psycopg2.connect(
    dbname="building",
    user="postgres",
    password="nina",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()
def clean_data(row):
    try:
        title = row[0].strip() if row[0] else "Untitled Event"
        name = row[1].strip() if row[1] else "Anonymous"
        location = row[3].strip() if row[3] else "Unknown Location"
        description = row[4].strip() if row[4] else "No description provided."
        try:
            event_date = datetime.strptime(row[2].strip(), "%Y-%m-%d").date()
        except ValueError:
            event_date = datetime.today().date()
        try:
            created_at = datetime.strptime(row[5].strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            created_at = datetime.now()
        return title, name, event_date, location, description, created_at
    except Exception as e:
        print(f"Error cleaning row {row}: {e}")
        return None
with open('events_dataset.csv', 'r', encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        cleaned_data = clean_data(row)
        if cleaned_data:
            cursor.execute(
                "SELECT id FROM employees WHERE id= %s AND name = %s AND position = %s AND hire_date=%s AND phone_number=  AND emergency_contact= AND email_address= AND user_id"  ,
                (cleaned_data[0], cleaned_data[1], cleaned_data[2])
            )
            existing_event = cursor.fetchone()
            if existing_event:
                cursor.execute(
                    """
                    UPDATE event_event
                    SET location = %s, description = %s, created_at = %s
                    WHERE id = %s
                    """,
                    (cleaned_data[3], cleaned_data[4], cleaned_data[5], existing_event[0])
                )
                print(f"Updated existing event: {cleaned_data[0]}")
            else:
                cursor.execute(
                    """
                    INSERT INTO event_event (title, name, date, location, description, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    cleaned_data
                )
                print(f"Inserted new event: {cleaned_data[0]}")
conn.commit()
cursor.close()
conn.close()
print("Data has been successfully processed and saved to the 'event_event' table.")