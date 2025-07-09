# Research Paper Analysis & Classification Pipeline

This repository contains a machine learning pipeline for analyzing research paper abstracts.
It performs two main tasks:

1.  **Classification:** Classifies abstracts into "Cancer" or "Non-Cancer" categories.

2.  **Disease Extraction:** Identifies specific disease names mentioned within the abstracts.

## Features

* **Data Preprocessing:** Handles PubMed abstract parsing, citation normalization, and missing data. Automatically infers labels from 'Cancer' and 'Non-Cancer' subdirectories.

* **Fine-tuned LLM for Classification:** Uses a `distilbert-base-uncased` model fine-tuned with **LoRA** for accurate text classification.

* **LLM for Disease Extraction:** Leverages `gemini-2.0-flash` via LangChain for robust disease entity recognition.

* **Performance Evaluation:** Provides accuracy, F1-score, and confusion matrices for model assessment, comparing fine-tuned performance against a baseline.

* **REST API:** Deploys the pipeline as a FastAPI service for easy integration.

* **Dockerization:** Containerizes the application for consistent deployment across environments.

* **Cloud Deployment Ready:** Includes a deployment script for Google Cloud Run and Hugging Face Spaces.

## Project Structure
```
.
├── app.py                      # FastAPI application for the pipeline
├── Dockerfile                  # Docker configuration for containerization
├── deploy.sh                   # Deployment script for local, GCP, and Hugging Face
├── main_pipeline.py            # Script for data preprocessing, training, and saving model
├── inference_and_evaluation.py # Script for loading saved model, evaluation, and local inference
├── results/                    # Directory for model checkpoints, logs, and performance reports
│   ├── fine_tuned_model/       # Saved fine-tuned model and tokenizer
│   ├── label_mappings.json     # JSON file mapping labels to IDs
│   └── performance_report.txt  # Detailed performance metrics
└── Dataset_1_/                 # Root directory for your dataset
    └── Dataset/
        ├── Cancer/             # Contains .txt files for Cancer abstracts
        │   ├── 
        ├── Non-Cancer/
            |── 
```
## Setup and Installation

### Prerequisites

* Python 3.9+

* `pip` (Python package installer)

* Docker (for containerization and deployment)

* Google Cloud SDK (for Google Cloud Run deployment)

* Git LFS and Hugging Face CLI (for Hugging Face Spaces deployment)

* Access to a Google Gemini API Key (for disease extraction) - Get one from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Local Setup

1.  **Clone the repository:**

    ```
    git clone <your-repo-link>
    cd <your-repo-name>
    ```

2.  **Create the dataset directory structure:**
    Organize your abstract `.txt` files as shown in the "Project Structure" section. For example, place your provided sample files like this:

    ```
    your-repo-name/
    └── Dataset_1_/
        └── Dataset/
            ├── Cancer/
            │   ├── 30878600.txt
            │   ├── 30879681.txt
            │   └── 30880225.txt
            └── Non-Cancer/
                ├── 23318425.txt
                ├── 24005976.txt
                └── 24075869.txt
    ```

3.  **Create a virtual environment (recommended):**

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    ```

4.  **Install dependencies:**

    ```
    pip install pandas numpy scikit-learn transformers==4.30.2 peft==0.4.0 torch==2.0.1 langchain-core langchain-google-genai==0.0.9 protobuf==3.20.3 grpcio==1.59.0 fastapi==0.103.2 "uvicorn[standard]==0.23.2" python-dotenv==1.0.0
    ```

5.  **Configure your Gemini API Key:**
    Open `main_pipeline.py` and `inference_and_evaluation.py` and replace `"YOUR_GEMINI_API_KEY"` with your actual API key. This key is crucial for the disease extraction functionality.

    ```
    API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY_HERE"
    ```

6.  **Run the training script (once):**
    This script performs data preprocessing, model fine-tuning, and saves the trained model and label mappings to the `./results/` directory.

    ```
    python main_pipeline.py
    ```

    *This step will take some time as it involves model training.*

7.  **Run the inference and evaluation script (multiple times):**
    After `main_pipeline.py` completes, you can run this script to load the saved model, evaluate its performance, and test inference without re-training.

    ```
    python inference_and_evaluation.py
    ```

## Model Selection & Justification

For this assignment, we use `distilbert-base-uncased` as the base small language model, fine-tuned with **LoRA (Low-Rank Adaptation)**.

* **DistilBERT:**

    * **Justification:** DistilBERT is a distilled version of BERT, making it smaller, faster, and lighter while retaining much of BERT's performance. This makes it an excellent choice for a "small language model" requirement, especially when computational resources or inference speed are concerns. It's pre-trained on a large corpus of English text, making it suitable for general text understanding.

* **LoRA (Low-Rank Adaptation):**

    * **Justification:** LoRA is a parameter-efficient fine-tuning technique. Instead of fine-tuning all parameters of the large pre-trained model, LoRA injects small, trainable rank decomposition matrices into existing layers. This significantly reduces the number of trainable parameters, leading to:

        * **Faster Training:** Fewer parameters to update.

        * **Reduced Memory Usage:** Less memory required during fine-tuning.

        * **Smaller Model Sizes:** The LoRA adapters are small and can be easily swapped, making deployment more flexible.

    * This approach is ideal for adapting a pre-trained model like DistilBERT to a specific downstream task (cancer/non-cancer classification) with limited domain-specific data, as it leverages the vast knowledge already encoded in the base model while efficiently learning task-specific features.

## Classification Task Performance

The pipeline evaluates the fine-tuned model's performance against a conceptual baseline provided in the assignment PDF.

### Baseline Model Performance (as per assignment PDF)

* **Accuracy:** 85%

* **F1-score:** 0.78

* **Confusion Matrix:**

    ```
                     Predicted Cancer | Predicted Non-Cancer
    Actual Cancer    | 320              | 80
    Actual Non-Cancer| 50               | 550
    ```

    *Total Samples: 1000*

### Fine-Tuned Model Performance

The `inference_and_evaluation.py` script will output the actual performance metrics of your fine-tuned model on its validation set. The confusion matrix will reflect the size of your validation set (e.g., 20% of your total dataset).

**Example Output (will vary based on your data and training run):**

--- Fine-Tuned Model Performance ---
Accuracy: 0.93
F1-score: 0.93

Confusion Matrix:
Cancer  Non-Cancer
Cancer          91           9
Non-Cancer       5          95

Classification Report:
precision    recall  f1-score   support

  Cancer       0.95      0.91      0.93       100
Non-Cancer       0.91      0.95      0.93       100

accuracy                           0.93       200
macro avg       0.93      0.93      0.93       200
weighted avg       0.93      0.93      0.93       200


**Performance Comparison:**
The fine-tuned model's performance (accuracy, F1-score) is directly compared to the baseline. A higher accuracy and F1-score for the fine-tuned model indicate successful performance improvement. The confusion matrix provides a detailed breakdown of true positives, true negatives, false positives, and false negatives, allowing for a deeper assessment of where the model excels or struggles. For instance, a reduction in "Actual Cancer" predicted as "Non-Cancer" (false negatives) is crucial for improving model reliability in a medical context.

## Running the API Locally with Docker

After running `main_pipeline.py` to train and save the model:

1.  **Build the Docker image:**

    ```
    docker build -t research-paper-classifier:latest .
    ```

2.  **Run the Docker container:**
    You will be prompted to enter your Gemini API Key.

    ```
    ./deploy.sh # Select option 3 for local Docker run
    ```

    Alternatively, you can run it manually and provide the API key:

    ```
    export GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY" # Replace with your actual key
    docker run -p 8000:8000 -e GEMINI_API_KEY="${GEMINI_API_KEY}" research-paper-classifier:latest
    ```

    The API will be accessible at `http://localhost:8000`.
    Access the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`.

## Cloud Deployment

The `deploy.sh` script provides options for deploying to Google Cloud Run or Hugging Face Spaces.

### Deploying to Google Cloud Run

1.  **Ensure Google Cloud SDK is installed and configured:**

    ```
    gcloud init
    gcloud auth login
    gcloud config set project YOUR_GCP_PROJECT_ID
    ```

2.  **Enable necessary APIs:**

    ```
    gcloud services enable run.googleapis.com artifactregistry.googleapis.com
    ```

3.  **Update `deploy.sh` placeholders:**
    Open `deploy.sh` and replace `your-gcp-project-id` and `us-central1` (region) with your actual project ID and desired region.

4.  **Run the deployment script:**

    ```
    ./deploy.sh
    ```

    Select `1` for Google Cloud Run. You will be prompted to enter your Gemini API Key.
    The script will build the Docker image, push it to Google Artifact Registry, and deploy it to Cloud Run.
    You can find your service URL in the `gcloud` output.

### Deploying to Hugging Face Spaces

1.  **Install Git LFS:**

    ```
    git lfs install
    ```

2.  **Install Hugging Face CLI:**

    ```
    pip install huggingface_hub
    ```

3.  **Log in to Hugging Face:**

    ```
    huggingface-cli login
    ```

4.  **Update `deploy.sh` placeholders:**
    Open `deploy.sh` and replace `your-huggingface-space-name` with your desired Space name.

5.  **Run the deployment script:**

    ```
    ./deploy.sh
    ```

    Select `2` for Hugging Face Spaces.
    The script will create a local Git repository, copy the necessary files (including the model), and push them to your Hugging Face Space.

    **Important for Hugging Face Spaces:**

    * After deployment, go to your Hugging Face Space settings and add your `GEMINI_API_KEY` as a secret. This is crucial for the disease extraction functionality.

    * The model weights (`fine_tuned_model` directory) will be tracked by Git LFS, so ensure Git LFS is correctly set up.

## API Usage

Once deployed, you can interact with the API:

### `POST /analyze_abstract`

Analyzes a single abstract.

**Request Body Example:**

{
    "abstract_id": "30878600",
    "abstract_text": "BACKGROUND: Tyrosine kinase inhibitors (TKIs) are clinically effective in non-small cell lung cancer (NSCLC) patients harbouring epidermal growth factor receptor (EGFR) oncogene mutations. Genetic factors, other than EGFR sensitive mutations, that allow prognosis of TKI treatment remain undefined. METHODS: We retrospectively screened 423 consecutive patients with advanced NSCLC and EGFR 19del or 21L858R mutations..."
}


**Response Body Example:**

{
    "classification": {
        "predicted_labels": ["Cancer"],
        "confidence_scores": {
            "Cancer": 0.98,
            "Non-Cancer": 0.02
        }
    },
    "disease_extraction": {
        "abstract_id": "30878600",
        "extracted_diseases": ["non-small cell lung cancer"]
    }
}


### `POST /batch_analyze_abstracts`

Analyzes multiple abstracts in a batch.

**Request Body Example:**
```
[
    {
        "abstract_id": "30878600",
        "abstract_text": "BACKGROUND: Tyrosine kinase inhibitors (TKIs) are clinically effective in non-small cell lung cancer (NSCLC) patients harbouring epidermal growth factor receptor (EGFR) oncogene mutations..."
    },
    {
        "abstract_id": "24005976",
        "abstract_text": "Psoriasis vulgaris is a genetically heterogenous disease with unclear molecular background. We assessed the association of psoriasis and its main clinical phenotypes..."
    }
]
```

**Response Body Example:**
```
[
    {
        "classification": {
            "predicted_labels": ["Cancer"],
            "confidence_scores": {
                "Cancer": 0.98,
                "Non-Cancer": 0.02
            }
        },
        "disease_extraction": {
            "abstract_id": "30878600",
            "extracted_diseases": ["non-small cell lung cancer"]
        }
    },
    {
        "classification": {
            "predicted_labels": ["Non-Cancer"],
            "confidence_scores": {
                "Cancer": 0.05,
                "Non-Cancer": 0.95
            }
        },
        "disease_extraction": {
            "abstract_id": "24005976",
            "extracted_diseases": ["Psoriasis vulgaris", "psoriasis"]
        }
    }
]
```

## Scalability Enhancements (Bonus)

* **Batch Processing:** The FastAPI application already includes a `/batch_analyze_abstracts` endpoint, allowing multiple abstracts to be processed in a single request, improving efficiency for larger inputs.

* **Streaming Capabilities:** For real-time, high-throughput scenarios, integrating with message brokers like Apache Kafka or Redis Streams would be beneficial.

    * A producer service would push new abstract data to a Kafka topic/Redis stream.

    * A consumer service (the FastAPI application, or a separate worker) would listen to these topics/streams, process the abstracts asynchronously, and push results to another topic/stream or a database.

    * This decouples ingestion from processing, enabling independent scaling of components and handling of backpressure.

* **Horizontal Scaling:** Cloud platforms like Google Cloud Run automatically scale the FastAPI application horizontally based on incoming traffic. For on-premise Docker deployments, container orchestration platforms like Kubernetes or Docker Swarm can be used to manage and scale multiple instances of the FastAPI container.

## Agentic Workflow and Orchestration (Bonus)

This pipeline can indeed be orchestrated as an agentic workflow solution, where different components act as specialized "agents" collaborating to achieve the overall goal.

1.  **Orchestrator Agent:** A central agent (e.g., built using frameworks like LangChain, CrewAI, or AutoGen) would receive the raw research papers or abstracts.

2.  **Preprocessing Agent/Tool:** The orchestrator would invoke a tool (which could be a separate microservice or a function within the orchestrator) to perform the data preprocessing steps (parsing, cleaning, citation normalization).

3.  **Classification Agent/Tool:** The orchestrator would then call the classification API endpoint (`/analyze_abstract` or `/batch_analyze_abstracts`) of our deployed FastAPI service to get the cancer/non-cancer classification and associated confidence scores.

4.  **Disease Identification Agent/Tool:** Concurrently or sequentially, the orchestrator would call the disease extraction part of the FastAPI API to identify specific disease names mentioned in the abstract.

5.  **Insight Generation Agent:** A more advanced agent could then take the classification, extracted diseases, and potentially other information (like key phrases, sentiment, or even a summary generated by another LLM) to synthesize deeper insights or generate a comprehensive analysis report for the research paper.

6.  **Output & Storage Agent:** Finally, an agent responsible for formatting and storing the results (e.g., saving to a database, generating a PDF report, or sending notifications).

This modular, agent-based approach allows for greater flexibility, maintainability, and the ability to easily swap out or add new capabilities (e.g., an agent for identifying drug interactions or clinical trial phases).