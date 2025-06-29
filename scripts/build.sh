#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setting up database..."
python setup.py

echo "Training AI model..."
python scripts/train_ai_system.py

echo "Build completed successfully!" 