#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setting up database..."
python setup.py

echo "Build completed successfully!" 