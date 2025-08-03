import mysql.connector

DB_CONFIG = {
    'user': 'root',
    'password': 'dbms',
    'host': 'localhost',
    'database': 'text_classification_db'
}

def write_and_move_predictions(predictions):
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # Update INSERT statement to include 'customer_id'
        insert_sql = "INSERT INTO labeled_queries (customer_id, query, category) VALUES (%s, %s, %s)"
        delete_sql = "DELETE FROM unlabeled_queries WHERE id = %s"
        
        # Adjust data to match the new format
        insert_data = [(p[1], p[2], p[3]) for p in predictions]
        delete_ids = [p[0] for p in predictions]

        cursor.executemany(insert_sql, insert_data)
        cursor.executemany(delete_sql, [(id,) for id in delete_ids])
        
        connection.commit()
        print(f"Inserted {len(predictions)} queries into 'labeled_queries'.")
        print(f"Removed {len(predictions)} queries from 'unlabeled_queries'.")
        
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        if connection:
            connection.rollback()
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()