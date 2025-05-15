#!/bin/bash
# Start script for the Axolotl GAN backend

# Ensure model files exist
bash check_model.sh

# Get port from environment or use default
PORT=${PORT:-5000}

echo "Starting Axolotl GAN backend on port $PORT"

# Start the application with gunicorn
gunicorn app:app --bind 0.0.0.0:$PORT --log-file -
