# Create a temporary container to copy out the database
docker create --name temp-sqlite jayelinjiaqi/my-sqlite-app:latest

# Copy bmarket.db from Docker image to runner filesystem
docker cp temp-sqlite:/data/bmarket.db ./data/bmarket.db

# Remove temporary container
docker rm temp-sqlite

# Run your ML pipeline script using the extracted database
python ml_pipeline.py
