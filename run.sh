#!/bin/bash

# Change to the zephyr-chat directory
cd api-demo-application

# Execute the streamlit command
streamlit run app.py &

# Change back to the previous directory
cd ..

# Execute the uvicorn command
uvicorn api:app --reload
