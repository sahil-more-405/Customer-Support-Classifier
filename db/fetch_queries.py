import mysql.connector

DB_CONFIG = {
    'user': 'root',
    'password': 'dbms',
    'host': 'localhost',
    'database': 'text_classification_db'
}

def fetch_unlabeled_queries():
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        # Fetching customer_id along with id and query
        cursor.execute("SELECT id, customer_id, query FROM unlabeled_queries")
        queries = cursor.fetchall()
        return queries
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()