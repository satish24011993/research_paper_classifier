#!/bin/bash

# Define variables
APP_NAME="research-paper-pipeline"
DOCKER_IMAGE_NAME="research-paper-classifier"
DOCKER_TAG="latest"
GCP_PROJECT_ID=79755680491 # REPLACE WITH YOUR GCP PROJECT ID
GCP_REGION="us-central1" # REPLACE WITH YOUR DESIRED GCP REGION
HF_SPACE_NAME="research_paper_classification" # REPLACE WITH YOUR HUGGING FACE SPACE NAME

# --- Step 1: Build Docker Image ---
echo "Building Docker image..."
docker build -t ${DOCKER_IMAGE_NAME}:${DOCKER_TAG} .
if [ $? -ne 0 ]; then
    echo "Docker image build failed."
    exit 1
fi
echo "Docker image built: ${DOCKER_IMAGE_NAME}:${DOCKER_TAG}"

# --- Step 2: Cloud Deployment Options ---

echo "Choose your deployment option:"
echo "1. Deploy to Google Cloud Run"
echo "2. Deploy to Hugging Face Spaces (requires Git LFS and HF CLI)"
echo "3. Run locally with Docker"
read -p "Enter your choice (1, 2, or 3): " choice

if [ "$choice" == "1" ]; then
    # --- Google Cloud Run Deployment ---
    echo "Deploying to Google Cloud Run..."

    # Authenticate Docker to Google Container Registry (GCR) or Artifact Registry (AR)
    # Using Artifact Registry (AR) as it's the recommended successor to GCR
    gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev
    if [ $? -ne 0 ]; then
        echo "gcloud docker authentication failed. Ensure gcloud CLI is installed and configured."
        exit 1
    fi

    # Tag the Docker image for Google Cloud Run
    docker tag ${DOCKER_IMAGE_NAME}:${DOCKER_TAG} ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${APP_NAME}/${DOCKER_IMAGE_NAME}:${DOCKER_TAG}
    if [ $? -ne 0 ]; then
        echo "Docker image tagging failed."
        exit 1
    fi

    # Push the Docker image to Google Artifact Registry
    docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${APP_NAME}/${DOCKER_IMAGE_NAME}:${DOCKER_TAG}
    if [ $? -ne 0 ]; then
        echo "Docker image push to Artifact Registry failed."
        exit 1
    fi
    echo "Docker image pushed to Google Artifact Registry."

    # Deploy to Google Cloud Run
    # The GEMINI_API_KEY should be passed as a secret or environment variable
    # For simplicity here, we pass it as an env var. In production, use --set-secrets.
    read -p "Enter your Gemini API Key for Cloud Run deployment: " GEMINI_API_KEY_INPUT
    gcloud run deploy ${APP_NAME} \
        --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${APP_NAME}/${DOCKER_IMAGE_NAME}:${DOCKER_TAG} \
        --platform managed \
        --region ${GCP_REGION} \
        --allow-unauthenticated \
        --port 8000 \
        --set-env-vars GEMINI_API_KEY="${GEMINI_API_KEY_INPUT}"
    if [ $? -ne 0 ]; then
        echo "Google Cloud Run deployment failed."
        exit 1
    fi
    echo "Deployment to Google Cloud Run successful!"
    echo "You can find your service URL in the gcloud output above."

elif [ "$choice" == "2" ]; then
    # --- Hugging Face Spaces Deployment ---
    echo "Deploying to Hugging Face Spaces..."
    echo "This requires 'git-lfs' and 'huggingface_hub' CLI to be installed and configured."
    echo "Ensure you have logged in to Hugging Face CLI: huggingface-cli login"

    # Create a new directory for the Hugging Face Space
    mkdir -p ${HF_SPACE_NAME}
    cd ${HF_SPACE_NAME}

    # Initialize a Git repository and link to Hugging Face Space
    git init
    git remote add origin https://huggingface.co/spaces/${HF_SPACE_NAME}
    git config --global user.email "you@example.com" # REPLACE WITH YOUR EMAIL
    git config --global user.name "Your Name" # REPLACE WITH YOUR NAME

    # Copy necessary files
    cp ../app.py .
    cp ../Dockerfile .
    # Copy the entire results directory which contains the fine_tuned_model and label_mappings.json
    cp -r ../results .

    # Create a .dockerignore file
    echo "__pycache__" > .dockerignore
    echo "*.pyc" >> .dockerignore
    echo ".git" >> .dockerignore
    echo ".gitignore" >> .dockerignore
    echo "Dataset_1_/" >> .dockerignore # Don't need the raw dataset in the deployed image

    # Create a README.md for the space
    echo "# ${HF_SPACE_NAME}" > README.md
    echo "" >> README.md
    echo "This is a Hugging Face Space for the Research Paper Analysis & Classification Pipeline." >> README.md
    echo "It uses a fine-tuned DistilBERT model for abstract classification and Gemini-2.0-flash for disease extraction." >> README.md
    echo "" >> README.md
    echo "## API Endpoints" >> README.md
    echo "- \`/analyze_abstract\`: Analyze a single abstract." >> README.md
    echo "- \`/batch_analyze_abstracts\`: Analyze multiple abstracts." >> README.md
    echo "- \`/health\`: Health check." >> README.md
    echo "" >> README.md
    echo "## Environment Variables" >> README.md
    echo "Set `GEMINI_API_KEY` in your Space secrets for disease extraction." >> README.md
    echo "" >> README.md
    echo "## Dockerfile" >> README.md
    echo "This Space is deployed using a Dockerfile." >> README.md

    # Add files to Git LFS if they are large (e.g., model weights)
    git lfs install
    git add .gitattributes # Ensure .gitattributes is added if LFS is used
    git add .
    git commit -m "Initial commit for Research Paper Analysis Pipeline"

    # Push to Hugging Face Space
    git push origin main
    if [ $? -ne 0 ]; then
        echo "Hugging Face Spaces deployment failed. Check your Git LFS setup and HF CLI login."
        exit 1
    fi
    echo "Deployment to Hugging Face Spaces successful!"
    echo "Your space should be available at https://huggingface.co/spaces/${HF_SPACE_NAME}"
    echo "Remember to set the GEMINI_API_KEY as a secret in your Hugging Face Space settings."
    cd .. # Go back to original directory

elif [ "$choice" == "3" ]; then
    # --- Local Docker Run ---
    echo "Running locally with Docker..."
    echo "This will expose the API on port 8000 on your local machine."
    read -p "Enter your Gemini API Key for local Docker run: " GEMINI_API_KEY_INPUT
    docker run -p 8000:8000 -e GEMINI_API_KEY="${GEMINI_API_KEY_INPUT}" ${DOCKER_IMAGE_NAME}:${DOCKER_TAG}
    if [ $? -ne 0 ]; then
        echo "Local Docker run failed."
        exit 1
    fi
    echo "FastAPI application is running locally on http://localhost:8000"
    echo "Access the API documentation at http://localhost:8000/docs"

else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo "Deployment script finished."