import pandas as pd
import numpy as np
import re
import os
import json
import torch # Still imported for potential torch-related utilities if needed, but not for model training directly.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # Requires xgboost library
from sklearn.pipeline import Pipeline # Useful for chaining vectorizer and classifier
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()
gemini_api_key = os.getenv('api_key')
# --- Configuration ---
# Output directory for models and results
OUTPUT_DIR = "./results"

# IMPORTANT: Replace "YOUR_GEMINI_API_KEY" with your actual Google Gemini API Key.
# You can get one from Google AI Studio: https://aistudio.google.com/app/apikey
API_KEY = gemini_api_key # <--- Update this line with your actual API Key!

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the base directory for your dataset
DATASET_BASE_DIR = "Dataset_1_/Dataset/"

# --- 1. Data Preprocessing ---

def parse_abstract_file(file_path):
    """Parses a single abstract file and extracts ID, Title, and Abstract."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    # Regex to find ID, Title, and Abstract
    match = re.search(r'<ID:(\d+)>[\s\n]*Title:([^\n]+)[\s\n]*Abstract:(.*)', content, re.DOTALL)
    if match:
        doc_id = match.group(1).strip()
        title = match.group(2).strip()
        abstract = match.group(3).strip()
        return {"id": doc_id, "title": title, "abstract": abstract}
    print(f"Warning: Could not parse abstract from {file_path}. Skipping.")
    return None

def load_and_preprocess_data(data_dir):
    """
    Loads and preprocesses abstract data from specified directory structure.
    Infers labels from subdirectory names (e.g., 'Cancer', 'Non-Cancer').
    Args:
        data_dir (str): Path to the root directory containing 'Cancer' and 'Non-Cancer' subfolders.
                        Example: "Dataset_1_/Dataset/"
    Returns:
        pd.DataFrame: A DataFrame with 'id', 'title', 'abstract', 'label', and 'label_id'.
        dict: Mapping from string labels to integer IDs.
        dict: Mapping from integer IDs to string labels.
    """
    all_data = []
    # Check if the base data directory exists
    if not os.path.isdir(data_dir):
        print(f"Error: Base dataset directory not found: {data_dir}. Please ensure it exists and contains 'Cancer' and 'Non-Cancer' subfolders.")
        return pd.DataFrame(), {}, {} # Return empty DataFrame and dicts

    for label_folder in ["Cancer", "Non-Cancer"]:
        current_label_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(current_label_path):
            print(f"Warning: Label directory not found: {current_label_path}. Skipping files from this category.")
            continue

        for filename in os.listdir(current_label_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(current_label_path, filename)
                parsed_data = parse_abstract_file(file_path)
                if parsed_data:
                    parsed_data["label"] = label_folder # Assign label based on folder name
                    all_data.append(parsed_data)

    df = pd.DataFrame(all_data)

    if df.empty:
        print("No data loaded. Please check your dataset path and file format.")
        return pd.DataFrame(), {}, {}

    # Handle missing abstracts (e.g., drop rows with empty abstracts)
    initial_rows = len(df)
    df.replace('', np.nan, inplace=True)
    df.dropna(subset=['abstract'], inplace=True)
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} rows with missing abstracts.")

    # Normalize citations (simple regex example, can be expanded)
    # This example removes common citation patterns like [1], [2,3], (Smith et al., 2020)
    df['abstract'] = df['abstract'].apply(lambda x: re.sub(r'\[\d+(?:,\d+)*\]|\([\w\s\d\.,]+\s+et al\.,\s+\d{4}\)', '', x))
    df['abstract'] = df['abstract'].apply(lambda x: re.sub(r'\s+', ' ', x).strip()) # Remove extra spaces

    # Map labels to integers for classification
    unique_labels = sorted(df['label'].unique())
    if not unique_labels:
        print("No unique labels found in the dataset. Cannot proceed with classification.")
        return pd.DataFrame(), {}, {}

    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for i, label in enumerate(unique_labels)}
    df['label_id'] = df['label'].map(label_to_id)
    print(f"Label mapping: {label_to_id}")
    return df, label_to_id, id_to_label

# --- 2. Model Training and Evaluation for Traditional ML Models ---

# Define compute metrics function at the global scope (for consistency)
def compute_metrics_traditional(true_labels, predictions, average='weighted'):
    """Computes accuracy, F1-score, and confusion matrix for traditional models."""
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average=average)
    cm = confusion_matrix(true_labels, predictions)
    return acc, f1, cm

def train_and_evaluate_traditional_models(df, label_to_id, id_to_label, output_dir=OUTPUT_DIR):
    """
    Trains and evaluates multiple traditional classification models.
    Args:
        df (pd.DataFrame): DataFrame with 'abstract' and 'label_id' columns.
        label_to_id (dict): Mapping from string labels to integer IDs.
        id_to_label (dict): Mapping from integer IDs to string labels.
        output_dir (str): Directory to save performance reports.
    Returns:
        dict: A dictionary of trained models.
        dict: A dictionary of performance metrics for each model.
    """
    trained_models = {}
    performance_metrics = {}

    # Split data
    if len(df) < 2:
        print("Not enough data for train/test split. Skipping training and evaluation.")
        return {}, {}

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['abstract'].tolist(), df['label_id'].tolist(), test_size=0.2, random_state=42, stratify=df['label_id']
    )

    # Define the vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    # Define models to test
    models_to_train = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "MultinomialNB": MultinomialNB(),
        "LinearSVC": LinearSVC(random_state=42, dual=False), # dual=False for large number of samples/features
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    print("\n--- Starting Training and Evaluation for Traditional ML Models ---")

    for model_name, model_instance in models_to_train.items():
        print(f"\nTraining {model_name}...")
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model_instance)
        ])

        try:
            pipeline.fit(train_texts, train_labels)
            predictions = pipeline.predict(val_texts)

            accuracy, f1, cm = compute_metrics_traditional(val_labels, predictions)
            report = classification_report(val_labels, predictions, target_names=[id_to_label[i] for i in sorted(id_to_label.keys())])

            trained_models[model_name] = pipeline
            performance_metrics[model_name] = {
                "accuracy": accuracy,
                "f1_score": f1,
                "confusion_matrix": cm.tolist(), # Convert numpy array to list for JSON serialization
                "classification_report": report
            }

            print(f"--- {model_name} Performance ---")
            print(f"Accuracy: {accuracy:.2f}")
            print(f"F1-score: {f1:.2f}")
            print("\nConfusion Matrix:")
            cm_df = pd.DataFrame(cm, index=[id_to_label[i] for i in sorted(id_to_label.keys())], columns=[id_to_label[i] for i in sorted(id_to_label.keys())])
            print(cm_df)
            print("\nClassification Report:")
            print(report)

        except Exception as e:
            print(f"Error training or evaluating {model_name}: {e}")
            performance_metrics[model_name] = {"error": str(e)}

    # Save performance metrics to a file
    with open(os.path.join(output_dir, "traditional_models_performance_report.json"), "w") as f:
        json.dump(performance_metrics, f, indent=4)
    print(f"Traditional models performance report saved to {os.path.join(output_dir, 'traditional_models_performance_report.json')}")

    print("\n--- Traditional ML Model Training and Evaluation Complete ---")
    return trained_models, performance_metrics, vectorizer, id_to_label # Return vectorizer and id_to_label for inference

# --- 3. Disease-Specific Identification from Abstract (using LangChain) ---

def extract_diseases_with_llm(abstract_text, api_key):
    """
    Extracts disease names from an abstract using an LLM.
    Args:
        abstract_text (str): The text of the abstract.
        api_key (str): The API key for the LLM.
    Returns:
        list: A list of extracted disease names.
    """
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("\nWARNING: Gemini API Key is not set or is the default placeholder. Disease extraction may fail.")
        print("Please replace 'YOUR_GEMINI_API_KEY' in the script with your actual API key.")
        return [] # Return empty list if API key is not set

    # Use gemini-2.0-flash by default.
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in biomedical text analysis. Your task is to accurately identify all specific disease names mentioned in the provided abstract. Return only a comma-separated list of disease names. If no diseases are mentioned, return 'None'."),
        ("user", "Abstract: {abstract}")
    ])

    chain = prompt_template | llm
    try:
        response = chain.invoke({"abstract": abstract_text})
        extracted_text = response.content.strip()
    except Exception as e:
        print(f"Error during LLM call for disease extraction: {e}")
        print("This could be due to an invalid API key, network issues, or rate limits.")
        return []

    if extracted_text.lower() == "none" or not extracted_text:
        return []
    return [disease.strip() for disease in extracted_text.split(',') if disease.strip()]

# --- Local Inference Functions for Traditional Models ---

def predict_single_abstract_traditional(abstract_text, model, vectorizer, id_to_label, api_key):
    """
    Performs classification and disease extraction for a single abstract using a traditional ML model.
    """
    # Classification
    # Transform the abstract text using the trained vectorizer
    features = vectorizer.transform([abstract_text])
    
    # Predict probabilities (if supported by the model) or decision function
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        predicted_id = np.argmax(probabilities)
        confidence_scores = {id_to_label[i]: prob for i, prob in enumerate(probabilities)}
    else: # For models like LinearSVC that don't have predict_proba
        predicted_id = model.predict(features)[0]
        # For models without predict_proba, confidence is harder to get directly.
        # We can use decision_function or assign a default confidence.
        if hasattr(model, 'decision_function'):
            decision_values = model.decision_function(features)[0]
            # For binary classification, convert decision_values to probabilities if needed
            # This is a simplification; for multi-class, it's more complex.
            if len(id_to_label) == 2:
                # Sigmoid for binary classification decision values
                prob_positive = 1 / (1 + np.exp(-decision_values))
                prob_negative = 1 - prob_positive
                confidence_scores = {id_to_label[1]: prob_positive, id_to_label[0]: prob_negative}
                # Ensure the predicted_id matches the highest probability
                predicted_id = np.argmax([prob_negative, prob_positive])
            else:
                # For multi-class decision_function, values are not probabilities directly.
                # A simple approach is to normalize them or just use raw decision values
                # as a proxy for confidence, or set a default.
                confidence_scores = {id_to_label[i]: val for i, val in enumerate(decision_values)}
        else:
            confidence_scores = {id_to_label[predicted_id]: 1.0} # Default to 1.0 confidence for the predicted class

    predicted_label = id_to_label.get(predicted_id, "Unknown")

    # Disease Extraction
    extracted_diseases = extract_diseases_with_llm(abstract_text, api_key)

    return {
        "classification": {
            "predicted_label": predicted_label,
            "confidence_scores": confidence_scores
        },
        "disease_extraction": {
            "extracted_diseases": extracted_diseases
        }
    }


if __name__ == "__main__":
    print("--- Starting Research Paper Analysis Pipeline (Local Test Mode) ---")

    # Load and preprocess data
    processed_df, label_to_id, id_to_label = load_and_preprocess_data(DATASET_BASE_DIR)

    if processed_df.empty:
        print("Exiting: No data available for training or testing.")
    else:
        # Train and evaluate traditional models
        trained_models, performance_metrics, vectorizer_trained, id_to_label_trained = \
            train_and_evaluate_traditional_models(processed_df, label_to_id, id_to_label)

        # Save label mappings for inference
        with open(os.path.join(OUTPUT_DIR, "label_mappings.json"), "w") as f:
            json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f)
        print(f"Label mappings saved to {os.path.join(OUTPUT_DIR, 'label_mappings.json')}")

        print("\n--- Local Inference Demonstration (using Logistic Regression as example) ---")
        # For demonstration, let's pick LogisticRegression for inference.
        # In a real scenario, you might pick the best performing model.
        if "LogisticRegression" in trained_models:
            inference_model = trained_models["LogisticRegression"]
            inference_vectorizer = vectorizer_trained # Use the vectorizer trained with the models

            # Example abstract for testing
            test_abstract_cancer = "BACKGROUND: Tyrosine kinase inhibitors (TKIs) are clinically effective in non-small cell lung cancer (NSCLC) patients harbouring epidermal growth factor receptor (EGFR) oncogene mutations. Genetic factors, other than EGFR sensitive mutations, that allow prognosis of TKI treatment remain undefined. METHODS: We retrospectively screened 423 consecutive patients with advanced NSCLC and EGFR 19del or 21L858R mutations."
            test_abstract_non_cancer = "Abstract: The hereditary autosomal recessive disease ataxia telangiectasia (A-T) is caused by mutation in the DNA damage kinase ATM. ATM's main function is to orchestrate DNA repair, thereby maintaining genomic stability. ATM activity is increased in response to several stimuli, including ionising radiation (IR) and hypotonic stress."

            print("\n--- Testing with a sample Cancer abstract ---")
            results_cancer = predict_single_abstract_traditional(test_abstract_cancer, inference_model, inference_vectorizer, id_to_label_trained, API_KEY)
            print(json.dumps(results_cancer, indent=2))

            print("\n--- Testing with a sample Non-Cancer abstract ---")
            results_non_cancer = predict_single_abstract_traditional(test_abstract_non_cancer, inference_model, inference_vectorizer, id_to_label_trained, API_KEY)
            print(json.dumps(results_non_cancer, indent=2))
        else:
            print("LogisticRegression model not trained or found. Cannot perform local inference demonstration.")

    print("\n--- Local Testing Complete ---")
