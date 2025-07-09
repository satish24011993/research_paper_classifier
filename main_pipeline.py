import pandas as pd
import numpy as np
import re
import os
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel # PeftModel is needed for loading in inference script
from torch.utils.data import Dataset
# LangChain imports are not strictly needed in this training-only script,
# but kept for consistency if you later integrate LLM calls directly here.
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from collections import defaultdict

# --- Configuration ---
# Set your Hugging Face model here. distilbert-base-uncased is a good small model.
MODEL_NAME = "distilbert-base-uncased"
# Set your output directory for models and results
OUTPUT_DIR = "./results"

# IMPORTANT: This API_KEY is primarily for the disease extraction part, which is now in the inference script.
# Keeping it here for completeness, but it's not used in this training-only script.
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('api_key') # <--- Update this line with your actual API Key!

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

# --- 2. Model Selection & Fine-tuning ---

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

def train_model(df, label_to_id, id_to_label, model_name=MODEL_NAME, output_dir=OUTPUT_DIR):
    """
    Trains and fine-tunes a classification model.
    Args:
        df (pd.DataFrame): DataFrame with 'abstract' and 'label_id' columns.
        label_to_id (dict): Mapping from string labels to integer IDs.
        id_to_label (dict): Mapping from integer IDs to string labels.
        model_name (str): Name of the pre-trained model from Hugging Face.
        output_dir (str): Directory to save the fine-tuned model.
    Returns:
        Trainer: The trained Hugging Face Trainer object.
        tokenizer: The tokenizer used for training.
        model: The fine-tuned model.
        list: Validation texts.
        list: Validation labels.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_to_id))

    # Define target modules for LoRA based on DistilBERT's architecture
    # These are typically the linear layers in the attention mechanism
    # You can inspect model.named_modules() to find appropriate layer names
    target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"] # For DistilBERTForSequenceClassification

    # Apply LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, # Sequence Classification
        inference_mode=False,
        r=8, # Rank
        lora_alpha=32, # Alpha parameter
        lora_dropout=0.1, # Dropout
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Split data
    if len(df) < 2: # Need at least 2 samples for train/test split
        print("Not enough data for train/test split. Skipping training.")
        return None, tokenizer, model, [], []

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['abstract'].tolist(), df['label_id'].tolist(), test_size=0.2, random_state=42, stratify=df['label_id']
    )

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    train_dataset = AbstractDataset(train_encodings, train_labels)
    val_dataset = AbstractDataset(val_encodings, val_labels)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "model_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none" # Disable integrations like wandb for simplicity
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer, # Pass tokenizer to trainer
        compute_metrics=compute_metrics, # Now globally defined
    )

    print("\n--- Starting Model Training ---")
    trainer.train()
    print("--- Model Training Complete ---")

    # Save the fine-tuned model
    model.save_pretrained(os.path.join(output_dir, "fine_tuned_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "fine_tuned_model"))
    print(f"Fine-tuned model saved to {os.path.join(output_dir, 'fine_tuned_model')}")

    return trainer, tokenizer, model, val_texts, val_labels # Return tokenizer here

# --- Main execution for training ---
if __name__ == "__main__":
    print("--- Starting Research Paper Analysis Pipeline (Training Mode) ---")

    # Load and preprocess data
    processed_df, label_to_id, id_to_label = load_and_preprocess_data(DATASET_BASE_DIR)

    if processed_df.empty:
        print("Exiting: No data available for training.")
    else:
        # Train the model
        trainer, tokenizer_trained, fine_tuned_model_trained, val_texts, val_labels = train_model(processed_df, label_to_id, id_to_label)

        # Save label mappings for inference (used by app.py and inference_and_evaluation.py)
        with open(os.path.join(OUTPUT_DIR, "label_mappings.json"), "w") as f:
            json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f)
        print(f"Label mappings saved to {os.path.join(OUTPUT_DIR, 'label_mappings.json')}")

        # The evaluation and local inference demonstration are moved to a separate script.
        print("\n--- Model Training Complete. Run 'inference_and_evaluation.py' for evaluation and inference. ---")

    