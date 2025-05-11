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
python src/train_model.py
