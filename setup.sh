#!/bin/bash

# Update and install system dependencies
echo "Updating system packages..."
apt-get update
apt-get install -y python3-pip python3-venv

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv /app/venv
source /app/venv/bin/activate

# Upgrade pip and install wheel
python -m pip install --upgrade pip wheel

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install Streamlit explicitly
pip install streamlit==1.48.1

# Make scripts executable
chmod +x setup.sh start.sh
