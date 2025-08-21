#!/bin/bash

# Install Python 3.10 if not already installed
if ! command -v python3.10 &> /dev/null; then
    echo "Installing Python 3.10..."
    apt-get update
    apt-get install -y python3.10 python3.10-venv
fi

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Make sure the script is executable
chmod +x setup.sh
