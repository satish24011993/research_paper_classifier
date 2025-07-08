# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
# This includes app.py, and the results/ directory which contains the trained model
COPY . /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Using specific versions for stability
RUN pip install --no-cache-dir \
    "transformers==4.30.2" \
    "peft==0.4.0" \
    "torch==2.0.1" \
    "pandas==2.0.3" \
    "scikit-learn==1.3.0" \
    "fastapi==0.103.2" \
    "uvicorn[standard]==0.23.2" \
    "langchain==0.0.300" \
    "langchain-google-genai==0.0.9" \
    "protobuf==3.20.3" \
    "grpcio==1.59.0" \
    "python-dotenv==1.0.0"

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
# Use 0.0.0.0 to make the server accessible from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
