# Connect to database
conn = sqlite3.connect("/data/bmarket.db")

# Fetch all rows from the 'lung_cancer' table
cursor = conn.cursor()
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
cursor.execute("SELECT * FROM bank_marketing")
rows = cursor.fetchall()

# Create DataFrame from the results
columns = [description[0] for description in cursor.description]  # Get column names
df = pd.DataFrame(rows, columns=columns)

# Print the DataFrame
print(df)

# Close the connection
conn.close()
