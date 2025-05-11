#!/bin/bash

# Debug
echo "Current directory: $(pwd)"
ls -al

# Ensure the local data directory exists
mkdir -p ./data

# Create temporary container from image
docker create --name temp-sqlite jayelinjiaqi/my-sqlite-app:latest

# Copy .db file from the Docker container to local data directory
docker cp temp-sqlite:/data/bmarket.db ./data/bmarket.db

# Clean up temporary container
docker rm temp-sqlite

# Run the ML pipeline
#python src/main.py

db_path='/home/runner/work/aiap20-jaye-lin-jiaqi-945I/aiap20-jaye-lin-jiaqi-945I/data/bmarket.db'
table_name='bank_marketing'

python src/train_model.py --db_path "$db_path" --table_name "$table_name"
