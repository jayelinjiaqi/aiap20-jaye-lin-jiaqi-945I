#!/bin/bash

# Debug
echo "=== Running pipeline ==="
echo "Current directory: $(pwd)"

# Ensure the local data directory exists
mkdir -p ./data

# Create temporary container from image
echo "Creating Docker container..."
docker create --name temp-sqlite jayelinjiaqi/my-sqlite-app:latest

# Copy .db file from the Docker container to local data directory
echo "Copying database from container..."
docker cp temp-sqlite:/data/bmarket.db ./data/bmarket.db

# Clean up temporary container
echo "Cleaning up Docker container..."
docker rm temp-sqlite

# Parameters to db_path and table name
db_path='/home/runner/work/aiap20-jaye-lin-jiaqi-945I/aiap20-jaye-lin-jiaqi-945I/data/bmarket.db'
table_name='bank_marketing'

# Run train_model.py in src folder
echo "Running model training..."
python src/train_model.py --db_path "$db_path" --table_name "$table_name"

echo "=== Pipeline completed successfully ==="
