import pandas as pd
import numpy as np
import re
import os
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import PeftModel, LoraConfig, TaskType # LoraConfig and TaskType might be needed if re-initializing model structure
from torch.utils.data import Dataset
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from collections import defaultdict

# --- Configuration ---
# Set your Hugging Face model here. distilbert-base-uncased is a good small model.
MODEL_NAME = "distilbert-base-uncased"
# Set your output directory where the trained model and mappings are saved
OUTPUT_DIR = "./results"


from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv('api_key')

# IMPORTANT: Replace "YOUR_GEMINI_API_KEY" with your actual Google Gemini API Key.
# You can get one from Google AI Studio: https://aistudio.google.com/app/apikey
API_KEY = GEMINI_API_KEY # <--- Update this line with your actual API Key!

# Define the base directory for your dataset (needed for re-creating val_texts)
DATASET_BASE_DIR = "Dataset_1_/Dataset/"

# --- Helper Functions (copied from main_pipeline.py for self-containment) ---

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

    match = re.search(r'<ID:(\d+)>[\s\n]*Title:([^\n]+)[\s\n]*Abstract:(.*)', content, re.DOTALL)
    if match:
        doc_id = match.group(1).strip()
        title = match.group(2).strip()
        abstract = match.group(3).strip()
        return {"id": doc_id, "title": title, "abstract": abstract}
    print(f"Warning: Could not parse abstract from {file_path}. Skipping.")
    return None

def load_and_preprocess_data_for_eval(data_dir):
    """
    Loads and preprocesses abstract data from specified directory structure.
    This is used to re-create the train/val split for evaluation.
    """
    all_data = []
    if not os.path.isdir(data_dir):
        print(f"Error: Base dataset directory not found: {data_dir}. Please ensure it exists.")
        return pd.DataFrame(), {}, {}

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
                    parsed_data["label"] = label_folder
                    all_data.append(parsed_data)

    df = pd.DataFrame(all_data)
    if df.empty:
        print("No data loaded for evaluation. Please check your dataset path and file format.")
        return pd.DataFrame(), {}, {}

    df.replace('', np.nan, inplace=True)
    df.dropna(subset=['abstract'], inplace=True)
    df['abstract'] = df['abstract'].apply(lambda x: re.sub(r'\[\d+(?:,\d+)*\]|\([\w\s\d\.,]+\s+et al\.,\s+\d{4}\)', '', x))
    df['abstract'] = df['abstract'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    unique_labels = sorted(df['label'].unique())
    if not unique_labels:
        print("No unique labels found in the dataset. Cannot proceed with classification.")
        return pd.DataFrame(), {}, {}
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    df['label_id'] = df['label'].map(label_to_id)

    return df, label_to_id, {i: label for label, i in label_to_id.items()}


class AbstractDataset(Dataset):
    """Custom Dataset for abstracts."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Define compute metrics function at the global scope
def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, predictions, average='weighted')
    acc = accuracy_score(p.label_ids, predictions)
    return {"accuracy": acc, "f1": f1}

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

def evaluate_model_performance(trainer, val_texts, val_labels, id_to_label, tokenizer, output_dir=OUTPUT_DIR):
    """
    Evaluates the fine-tuned model's performance and prints metrics.
    Args:
        trainer (Trainer): The trained Hugging Face Trainer.
        val_texts (list): List of validation abstract texts.
        val_labels (list): List of true validation labels (integer IDs).
        id_to_label (dict): Mapping from integer IDs to string labels.
        tokenizer: The tokenizer used for encoding.
        output_dir (str): Directory to save performance reports.
    """
    if trainer is None or not val_texts:
        print("\nSkipping fine-tuned model evaluation: Trainer not available or no validation data.")
        return

    # Evaluate baseline (conceptual, as per assignment PDF)
    print("\n--- Baseline Model Performance (as per assignment PDF) ---")
    print("Accuracy: 85%")
    print("F1-score: 0.78")
    print("Confusion Matrix:")
    print("                 Predicted Cancer | Predicted Non-Cancer")
    print("Actual Cancer    | 320              | 80")
    print("Actual Non-Cancer| 50               | 550")

    # Evaluate fine-tuned model
    predictions = trainer.predict(AbstractDataset(tokenizer(val_texts, truncation=True, padding=True, max_length=512), val_labels))
    preds = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    # Calculate metrics
    accuracy = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')
    cm = confusion_matrix(true_labels, preds)
    report = classification_report(true_labels, preds, target_names=[id_to_label[i] for i in sorted(id_to_label.keys())])

    print("\n--- Fine-Tuned Model Performance ---")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1-score: {f1:.2f}")
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=[id_to_label[i] for i in sorted(id_to_label.keys())], columns=[id_to_label[i] for i in sorted(id_to_label.keys())])
    print(cm_df)
    print("\nClassification Report:")
    print(report)

    # Performance Improvement Analysis
    print("\n--- Performance Improvement Analysis ---")
    print(f"Accuracy increased by {(accuracy - 0.85)*100:.2f}% after fine-tuning (compared to baseline 85%).")
    print(f"F1-score increased by {(f1 - 0.78)*100:.2f} after fine-tuning (compared to baseline 0.78).")
    print("Further analysis on false negatives/positives can be done by inspecting the confusion matrix.")

    # Save performance metrics to a file
    with open(os.path.join(output_dir, "performance_report.txt"), "w") as f:
        f.write("--- Baseline Model Performance (as per assignment PDF) ---\n")
        f.write("Accuracy: 85%\n")
        f.write("F1-score: 0.78\n")
        f.write("Confusion Matrix:\n")
        f.write("                 Predicted Cancer | Predicted Non-Cancer\n")
        f.write("Actual Cancer    | 320              | 80\n")
        f.write("Actual Non-Cancer| 50               | 550\n\n")
        f.write("--- Fine-Tuned Model Performance ---\n")
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"F1-score: {f1:.2f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n\nClassification Report:\n")
        f.write(report)
        f.write("\n\n--- Performance Improvement Analysis ---\n")
        f.write(f"Accuracy increased by {accuracy - 0.85:.2f}% after fine-tuning (compared to baseline 85%).\n")
        f.write(f"F1-score increased by {f1 - 0.78:.2f} after fine-tuning (compared to baseline 0.78).\n")
        f.write("Further analysis on false negatives/positives can be done by inspecting the confusion matrix.\n")
    print(f"Performance report saved to {os.path.join(output_dir, 'performance_report.txt')}")

def load_model_for_inference(model_path, base_model_name, num_labels):
    """Loads the fine-tuned model and tokenizer for inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval() # Set model to evaluation mode
    return tokenizer, model

def predict_single_abstract(abstract_text, tokenizer, model, id_to_label, api_key):
    """
    Performs classification and disease extraction for a single abstract.
    """
    # Classification
    inputs = tokenizer(abstract_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0]

    predicted_id = torch.argmax(probabilities).item()
    predicted_label = id_to_label.get(predicted_id, "Unknown")
    confidence_scores = {id_to_label[i]: prob.item() for i, prob in enumerate(probabilities)}

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
    print("--- Starting Research Paper Analysis Pipeline (Inference & Evaluation Mode) ---")

    # Load label mappings
    LABEL_MAPPINGS_PATH = os.path.join(OUTPUT_DIR, "label_mappings.json")
    if not os.path.exists(LABEL_MAPPINGS_PATH):
        print(f"Error: Label mappings file not found at {LABEL_MAPPINGS_PATH}.")
        print("Please run 'main_pipeline.py' first to train the model and generate necessary files.")
        exit()

    with open(LABEL_MAPPINGS_PATH, "r") as f:
        label_mappings = json.load(f)
    label_to_id = label_mappings["label_to_id"]
    id_to_label = {int(k): v for k, v in label_mappings["id_to_label"].items()}
    num_labels_loaded = len(id_to_label)
    print("Label mappings loaded successfully.")

    # Load the model for inference and evaluation
    MODEL_PATH = os.path.join(OUTPUT_DIR, "fine_tuned_model")
    try:
        inference_tokenizer, inference_model = load_model_for_inference(
            MODEL_PATH,
            MODEL_NAME,
            num_labels_loaded
        )
        print("Model loaded successfully for inference and evaluation.")
    except Exception as e:
        print(f"Error loading model for inference/evaluation: {e}.")
        print("Please ensure 'main_pipeline.py' was run successfully and the model is saved correctly.")
        exit()

    # Re-load data to get val_texts and val_labels for evaluation
    # This ensures the split is consistent with training if random_state is fixed.
    processed_df_for_eval, _, _ = load_and_preprocess_data_for_eval(DATASET_BASE_DIR)
    if processed_df_for_eval.empty:
        print("Exiting: No data available for evaluation.")
        exit()

    # Re-create train/val split to get the exact val_texts and val_labels used during training
    _, val_texts_for_eval, _, val_labels_for_eval = train_test_split(
        processed_df_for_eval['abstract'].tolist(), processed_df_for_eval['label_id'].tolist(),
        test_size=0.2, random_state=42, stratify=processed_df_for_eval['label_id']
    )

    # Create a dummy trainer for evaluation purposes (it only needs the model, tokenizer, and eval_dataset)
    # We don't need to train it, just use its predict method.
    dummy_trainer = Trainer(
        model=inference_model,
        args=TrainingArguments(output_dir="./tmp_eval", report_to="none"), # Dummy output dir
        eval_dataset=AbstractDataset(inference_tokenizer(val_texts_for_eval, truncation=True, padding=True, max_length=512), val_labels_for_eval),
        tokenizer=inference_tokenizer,
        compute_metrics=compute_metrics,
    )

    # Evaluate model performance
    evaluate_model_performance(dummy_trainer, val_texts_for_eval, val_labels_for_eval, id_to_label, inference_tokenizer)

    print("\n--- Local Inference Demonstration ---")
    # Example abstract for testing
    test_abstract_cancer = "BACKGROUND: Tyrosine kinase inhibitors (TKIs) are clinically effective in non-small cell lung cancer (NSCLC) patients harbouring epidermal growth factor receptor (EGFR) oncogene mutations. Genetic factors, other than EGFR sensitive mutations, that allow prognosis of TKI treatment remain undefined. METHODS: We retrospectively screened 423 consecutive patients with advanced NSCLC and EGFR 19del or 21L858R mutations."
    test_abstract_non_cancer = "Abstract: The hereditary autosomal recessive disease ataxia telangiectasia (A-T) is caused by mutation in the DNA damage kinase ATM. ATM's main function is to orchestrate DNA repair, thereby maintaining genomic stability. ATM activity is increased in response to several stimuli, including ionising radiation (IR) and hypotonic stress."

    print("\n--- Testing with a sample Cancer abstract ---")
    results_cancer = predict_single_abstract(test_abstract_cancer, inference_tokenizer, inference_model, id_to_label, API_KEY)
    print(json.dumps(results_cancer, indent=2))

    print("\n--- Testing with a sample Non-Cancer abstract ---")
    results_non_cancer = predict_single_abstract(test_abstract_non_cancer, inference_tokenizer, inference_model, id_to_label, API_KEY)
    print(json.dumps(results_non_cancer, indent=2))

    print("\n--- Inference & Evaluation Complete ---")
