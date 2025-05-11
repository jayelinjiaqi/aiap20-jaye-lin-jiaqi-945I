import sqlite3
import pandas as pd

def load_data_from_db(db_path, table_name):
    """
    Connects to the SQLite database and loads data from a specified table.
    
    Args:
    - db_path: Path to the SQLite database.
    - table_name: Name of the table to load.
    
    Returns:
    - df: Pandas DataFrame containing the data from the specified table.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Fetch all rows from the specified table
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    
    # Create DataFrame from the results
    columns = [description[0] for description in cursor.description]  # Get column names
    df = pd.DataFrame(rows, columns=columns)
    
    # Close the connection
    conn.close()
    
    return df
