from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import re
import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv('api_key')

# Initialize FastAPI app
app = FastAPI(
    title="Research Paper Analysis & Classification Pipeline",
    description="API for classifying research paper abstracts and extracting diseases.",
    version="1.0.0"
)

# --- Configuration ---
MODEL_PATH = "./results/fine_tuned_model" # Path where the fine-tuned model is saved
LABEL_MAPPINGS_PATH = "./results/label_mappings.json"
# API key for the LLM (loaded from environment variables in production, or fallback)
LLM_API_KEY = GEMINI_API_KEY # Will be set via environment variable in Docker/Cloud
MODEL_NAME_FOR_API = "distilbert-base-uncased" # Base model name for loading

# Global variables for model, tokenizer, and label mappings
tokenizer = None
model = None
id_to_label = {}
label_to_id = {}
llm_api_instance = None

# --- Initialization Function (runs once when app starts) ---
@app.on_event("startup")
async def load_resources():
    global tokenizer, model, id_to_label, label_to_id, llm_api_instance

    print("Attempting to load model and resources...")
    # Load label mappings first to get num_labels
    try:
        with open(LABEL_MAPPINGS_PATH, "r") as f:
            label_mappings = json.load(f)
        id_to_label = {int(k): v for k, v in label_mappings["id_to_label"].items()}
        label_to_id = label_mappings["label_to_id"]
        num_labels_loaded = len(id_to_label)
        print(f"Label mappings loaded successfully: {id_to_label}")
    except FileNotFoundError:
        print(f"Error: Label mappings file not found at {LABEL_MAPPINGS_PATH}. "
              "Please ensure 'main_pipeline.py' was run successfully to generate it.")
        # Fallback to default if mappings cannot be loaded
        id_to_label = {0: "Cancer", 1: "Non-Cancer"}
        label_to_id = {"Cancer": 0, "Non-Cancer": 1}
        num_labels_loaded = 2
        print("Using fallback label mappings for API.")
    except Exception as e:
        print(f"Error loading label mappings for API: {e}")
        id_to_label = {0: "Cancer", 1: "Non-Cancer"}
        label_to_id = {"Cancer": 0, "Non-Cancer": 1}
        num_labels_loaded = 2
        print("Using fallback label mappings for API due to error.")

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME_FOR_API,
            num_labels=num_labels_loaded
        )
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model.eval() # Set model to evaluation mode
        print("Classification model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading classification model or tokenizer: {e}")
        tokenizer = None
        model = None
        print("Classification model will not be available.")

    # Initialize LLM for disease extraction
    try:
        if LLM_API_KEY: # Only initialize if API_KEY is provided
            llm_api_instance = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=LLM_API_KEY)
            print("Gemini LLM initialized for disease extraction.")
        else:
            print("WARNING: GEMINI_API_KEY environment variable not set. Disease extraction will not function.")
    except Exception as e:
        print(f"Error initializing Gemini LLM in API: {e}")
        llm_api_instance = None
        print("Disease extraction will not be available.")


# --- Request and Response Models ---
class AbstractInput(BaseModel):
    abstract_id: str
    abstract_text: str

class ClassificationOutput(BaseModel):
    predicted_labels: list[str]
    confidence_scores: dict[str, float]

class DiseaseExtractionOutput(BaseModel):
    abstract_id: str
    extracted_diseases: list[str]

class PipelineOutput(BaseModel):
    classification: ClassificationOutput
    disease_extraction: DiseaseExtractionOutput

# --- Helper Functions ---
def preprocess_text_for_api(text: str) -> str:
    """Basic text preprocessing for API input."""
    text = re.sub(r'\[\d+(?:,\d+)*\]|\([\w\s\d\.,]+\s+et al\.,\s+\d{4}\)', '', text) # Remove citations
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

async def classify_abstract_api(abstract_text: str) -> ClassificationOutput:
    """Classifies an abstract into cancer/non-cancer categories."""
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded. Please check server logs.")

    preprocessed_text = preprocess_text_for_api(abstract_text)
    inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0] # Get probabilities for the first (and only) item in batch

    # Get predicted label
    predicted_id = torch.argmax(probabilities).item()
    predicted_label = id_to_label.get(predicted_id, "Unknown")

    # Get confidence scores for all labels
    confidence_scores = {id_to_label[i]: prob.item() for i, prob in enumerate(probabilities)}

    return ClassificationOutput(
        predicted_labels=[predicted_label],
        confidence_scores=confidence_scores
    )

async def extract_diseases_api(abstract_id: str, abstract_text: str) -> DiseaseExtractionOutput:
    """Extracts diseases from an abstract using the LLM."""
    if llm_api_instance is None:
        return DiseaseExtractionOutput(abstract_id=abstract_id, extracted_diseases=["Error: Gemini API Key not configured or LLM failed to initialize."])

    chain = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in biomedical text analysis. Your task is to accurately identify all specific disease names mentioned in the provided abstract. Return only a comma-separated list of disease names. If no diseases are mentioned, return 'None'."),
        ("user", "Abstract: {abstract}")
    ]) | llm_api_instance # Use the global llm_api_instance

    try:
        response = await chain.ainvoke({"abstract": abstract_text})
        extracted_text = response.content.strip()
        if extracted_text.lower() == "none" or not extracted_text:
            diseases = []
        else:
            diseases = [disease.strip() for disease in extracted_text.split(',') if disease.strip()]
        return DiseaseExtractionOutput(abstract_id=abstract_id, extracted_diseases=diseases)
    except Exception as e:
        print(f"Error during disease extraction for {abstract_id}: {e}")
        return DiseaseExtractionOutput(abstract_id=abstract_id, extracted_diseases=[f"Error extracting diseases: {e}"])

# --- API Endpoints ---
@app.post("/analyze_abstract", response_model=PipelineOutput)
async def analyze_abstract_endpoint(input: AbstractInput):
    """
    Analyzes a single research paper abstract, performing classification and disease extraction.
    """
    classification_result = await classify_abstract_api(input.abstract_text)
    disease_extraction_result = await extract_diseases_api(input.abstract_id, input.abstract_text)

    return PipelineOutput(
        classification=classification_result,
        disease_extraction=disease_extraction_result
    )

@app.post("/batch_analyze_abstracts", response_model=list[PipelineOutput])
async def batch_analyze_abstracts_endpoint(inputs: list[AbstractInput]):
    """
    Analyzes multiple research paper abstracts in a batch.
    """
    results = []
    for input_item in inputs:
        classification_result = await classify_abstract_api(input_item.abstract_text)
        disease_extraction_result = await extract_diseases_api(input_item.abstract_id, input_item.abstract_text)
        results.append(PipelineOutput(
            classification=classification_result,
            disease_extraction=disease_extraction_result
        ))
    return results

@app.get("/health")
async def health_check():
    return {"status": "ok", "classification_model_loaded": True if model is not None else False, "llm_for_disease_extraction_loaded": True if llm_api_instance is not None else False}
