import mysql.connector
import csv

# Database configuration
DB_NAME = 'text_classification_db'
DB_CONFIG = {
    'user': 'root',
    'password': 'dbms',
    'host': 'localhost',
}

# CSV file paths
LABELED_DATA_CSV = 'data/labeled_data.csv'
UNLABELED_DATA_CSV = 'data/unlabeled_data.csv'

def read_labeled_csv_data(file_path):
    data = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) == 3:
                    data.append((int(row[0]), row[1], row[2]))
                else:
                    print(f"Skipping malformed row in '{file_path}': {row}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}")
        return None
    return data

def read_unlabeled_csv_data(file_path):
    data = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) == 2:
                    data.append((int(row[0]), row[1]))
                else:
                    print(f"Skipping malformed row in '{file_path}': {row}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}")
        return None
    return data

try:
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor()

    cursor.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
    cursor.execute(f"CREATE DATABASE {DB_NAME}")
    cursor.execute(f"USE {DB_NAME}")
    print(f"Database '{DB_NAME}' has been recreated.")

    table_name_labeled = 'labeled_queries'
    print(f"Creating table '{table_name_labeled}'... ", end='')
    table_description_labeled = f"""
    CREATE TABLE {table_name_labeled} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        customer_id INT NOT NULL,
        query TEXT NOT NULL,
        category VARCHAR(50) NOT NULL
    );
    """
    cursor.execute(table_description_labeled)
    print("OK")

    table_name_unlabeled = 'unlabeled_queries'
    print(f"Creating table '{table_name_unlabeled}'... ", end='')
    table_description_unlabeled = f"""
    CREATE TABLE {table_name_unlabeled} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        customer_id INT NOT NULL,
        query TEXT NOT NULL
    );
    """
    cursor.execute(table_description_unlabeled)
    print("OK")
    print("\nSchema creation process completed.")

    queries_labeled = read_labeled_csv_data(LABELED_DATA_CSV)
    new_queries = read_unlabeled_csv_data(UNLABELED_DATA_CSV)
    
    if queries_labeled is not None and new_queries is not None:
        insert_labeled_query = f"INSERT INTO {table_name_labeled} (customer_id, query, category) VALUES (%s, %s, %s)"
        print("\nInserting labeled data...")
        cursor.executemany(insert_labeled_query, queries_labeled)
        connection.commit()
        print(f"Inserted {cursor.rowcount} labeled queries.")

        insert_unlabeled_query = f"INSERT INTO {table_name_unlabeled} (customer_id, query) VALUES (%s, %s)"
        print("Inserting unlabeled queries...")
        cursor.executemany(insert_unlabeled_query, new_queries)
        connection.commit()
        print(f"Inserted {cursor.rowcount} unlabeled queries.")

except mysql.connector.Error as err:
    print(f"Error: {err}")
    
finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("\nMySQL connection is closed.")