#!/bin/bash

# Define variables
APP_NAME="research-paper-pipeline"
DOCKER_IMAGE_NAME="research-paper-classifier"
DOCKER_TAG="latest"
GCP_PROJECT_ID=79755680491 # REPLACE WITH YOUR GCP PROJECT ID
GCP_REGION="us-central1" # REPLACE WITH YOUR DESIRED GCP REGION
HF_SPACE_NAME="reasearch-paper-classifier-space" # REPLACE WITH YOUR HUGGING FACE SPACE NAME

# --- Step 0: Initial Setup (Optional, run once if needed) ---
# Uncomment and run these lines if you need to set up Git user info for Hugging Face
# git config --global user.email "you@example.com" # REPLACE WITH YOUR EMAIL
# git config --global user.name "Your Name" # REPLACE WITH YOUR NAME
# huggingface-cli login # Run this to log in to Hugging Face CLI

# --- Main Menu ---
echo "Choose an action:"
echo "1. Build Docker Image"
echo "2. Run Local Docker Container (requires image to be built)"
echo "3. Deploy to Google Cloud Run (requires image to be built and pushed)"
echo "4. Deploy to Hugging Face Spaces (requires image to be built locally for verification, but HF builds from Dockerfile)"
echo "5. Exit"
read -p "Enter your choice (1, 2, 3, 4, or 5): " choice

if [ "$choice" == "1" ]; then
    # --- Build Docker Image ---
    echo "Building Docker image..."
    docker build -t ${DOCKER_IMAGE_NAME}:${DOCKER_TAG} .
    if [ $? -ne 0 ]; then
        echo "Docker image build failed."
        exit 1
    fi
    echo "Docker image built: ${DOCKER_IMAGE_NAME}:${DOCKER_TAG}"
    echo "You can now run the container locally or deploy it."

elif [ "$choice" == "2" ]; then
    # --- Run Local Docker Container ---
    echo "Running locally with Docker..."
    echo "This will expose the API on port 8000 on your local machine."
    # Check if image exists
    if ! docker images -q ${DOCKER_IMAGE_NAME}:${DOCKER_TAG} | grep -q .; then
        echo "Error: Docker image '${DOCKER_IMAGE_NAME}:${DOCKER_TAG}' not found."
        echo "Please build the image first by selecting option 1."
        exit 1
    fi
    read -p "Enter your Gemini API Key for local Docker run: " GOOGLE_API_KEY_INPUT
    docker run -p 8000:8000 -e GOOGLE_API_KEY="${GOOGLE_API_KEY_INPUT}" ${DOCKER_IMAGE_NAME}:${DOCKER_TAG}
    if [ $? -ne 0 ]; then
        echo "Local Docker run failed."
        exit 1
    fi
    echo "FastAPI application is running locally on http://localhost:8000"
    echo "Access the API documentation at http://localhost:8000/docs"

elif [ "$choice" == "3" ]; then
    # --- Google Cloud Run Deployment ---
    echo "Deploying to Google Cloud Run..."

    # Check if image exists locally before pushing
    if ! docker images -q ${DOCKER_IMAGE_NAME}:${DOCKER_TAG} | grep -q .; then
        echo "Error: Docker image '${DOCKER_IMAGE_NAME}:${DOCKER_TAG}' not found locally."
        echo "Please build the image first by selecting option 1."
        exit 1
    fi

    # Authenticate Docker to Google Artifact Registry (AR)
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
    read -p "Enter your Gemini API Key for Cloud Run deployment: " GOOGLE_API_KEY_INPUT
    gcloud run deploy ${APP_NAME} \
        --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${APP_NAME}/${DOCKER_IMAGE_NAME}:${DOCKER_TAG} \
        --platform managed \
        --region ${GCP_REGION} \
        --allow-unauthenticated \
        --port 8000 \
        --set-env-vars GOOGLE_API_KEY="${GOOGLE_API_KEY_INPUT}"
    if [ $? -ne 0 ]; then
        echo "Google Cloud Run deployment failed."
        exit 1
    fi
    echo "Deployment to Google Cloud Run successful!"
    echo "You can find your service URL in the gcloud output above."

elif [ "$choice" == "4" ]; then
    # --- Hugging Face Spaces Deployment ---
    echo "Deploying to Hugging Face Spaces..."
    echo "This requires 'git-lfs' and 'huggingface_hub' CLI to be installed and configured."
    echo "Ensure you have logged in to Hugging Face CLI: huggingface-cli login"
    echo "Also, ensure you have manually created the Hugging Face Space named '${HF_SPACE_NAME}' with 'Docker' SDK."

    # Create a new directory for the Hugging Face Space
    mkdir -p ${HF_SPACE_NAME}
    cd ${HF_SPACE_NAME}

    # Initialize a Git repository
    git init
    git branch -M main # Renames current branch to 'main' or creates it if it doesn't exist
    git remote add origin https://huggingface.co/spaces/${HF_SPACE_NAME}
    # Ensure user.email and user.name are configured globally or uncomment and set them here
    # git config user.email "your@email.com"
    # git config user.name "Your Name"

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
    echo "Set \`GOOGLE_API_KEY\` in your Space secrets for disease extraction." >> README.md
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
    echo "Remember to set the GOOGLE_API_KEY as a secret in your Hugging Face Space settings."
    cd .. # Go back to original directory
# Or else do manually by cloning the repo
elif [ "$choice" == "5" ]; then
    echo "Exiting."
    exit 0

else
    echo "Invalid choice. Please enter 1, 2, 3, 4, or 5."
    exit 1
fi

echo "Deployment script finished."
