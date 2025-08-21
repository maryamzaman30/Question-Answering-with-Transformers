#!/bin/bash

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true

# Create necessary directories
mkdir -p ~/.streamlit/

# Create config file
echo "[server]" > ~/.streamlit/config.toml
echo "headless = true" >> ~/.streamlit/config.toml
echo "address = '0.0.0.0'" >> ~/.streamlit/config.toml
echo "port = 8501" >> ~/.streamlit/config.toml
echo "enableCORS = false" >> ~/.streamlit/config.toml
echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml

# Start Streamlit
exec streamlit run app.py
