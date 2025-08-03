import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import os
import json
import logging
from typing import List, Tuple, Dict, Any

# --- Configuration ---
class Settings:
    # Database credentials should ideally be loaded from environment variables
    # for better security, e.g., using os.getenv('DB_USER', 'root')
    DB_USER = 'root'
    DB_PASSWORD = 'dbms'
    DB_HOST = 'localhost'
    DB_NAME = 'text_classification_db'
    
    # Model & Data Configuration
    BASE_MODEL_NAME = "bert-base-uncased"
    SAVE_PATH = "./model/fine_tuned_model"
    LOG_LEVEL = "INFO"

    @property
    def labels_file_path(self) -> str:
        return os.path.join(self.SAVE_PATH, "labels.json")

settings = Settings()

# --- Custom Dataset Class ---
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# --- Helper Functions ---
def setup_logging():
    """Configures the logger for the script."""
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def fetch_data() -> pd.DataFrame:
    """Fetches labeled query data from the MySQL database."""
    db_config = {
        'user': settings.DB_USER,
        'password': settings.DB_PASSWORD,
        'host': settings.DB_HOST,
        'database': settings.DB_NAME
    }
    connection = None
    try:
        logger.info(f"Connecting to database '{settings.DB_NAME}' on host '{settings.DB_HOST}'...")
        connection = mysql.connector.connect(**db_config)
        query = "SELECT query, category FROM labeled_queries"
        df = pd.read_sql(query, connection)
        logger.info(f"Successfully fetched {len(df)} records from the database.")
        return df
    except mysql.connector.Error as err:
        logger.error(f"Error fetching data from database: {err}")
        return pd.DataFrame()
    finally:
        if connection and connection.is_connected():
            connection.close()
            logger.info("Database connection closed.")

def prepare_datasets(df: pd.DataFrame, tokenizer: BertTokenizerFast) -> Tuple[TextDataset, TextDataset, Dict[str, int]]:
    """Prepares training and validation datasets from the dataframe."""
    unique_categories = sorted(df['category'].unique().tolist())
    label_to_id = {label: i for i, label in enumerate(unique_categories)}

    logger.info(f"Found {len(unique_categories)} unique labels: {unique_categories}")

    df['labels'] = df['category'].map(label_to_id)
    df.dropna(subset=['labels'], inplace=True)
    df['labels'] = df['labels'].astype(int)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['query'].tolist(), df['labels'].tolist(), test_size=0.2, stratify=df['labels'], random_state=42
    )

    logger.info(f"Splitting data: {len(train_texts)} training samples, {len(val_texts)} validation samples.")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)

    return train_dataset, val_dataset, label_to_id

def train_model(train_dataset: TextDataset, val_dataset: TextDataset, num_labels: int):
    """Initializes and runs the Hugging Face Trainer."""
    logger.info("Loading pre-trained model for sequence classification...")
    model = BertForSequenceClassification.from_pretrained(settings.BASE_MODEL_NAME, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info("Starting model fine-tuning...")
    trainer.train()
    logger.info("Fine-tuning complete.")

    logger.info(f"Saving the best model to {settings.SAVE_PATH}")
    trainer.save_model(settings.SAVE_PATH)

def main():
    """Main function to orchestrate the data fetching, preparation, and model training."""
    df = fetch_data()
    if df.empty:
        logger.error("No data fetched from the database. Aborting training.")
        return

    tokenizer = BertTokenizerFast.from_pretrained(settings.BASE_MODEL_NAME)
    train_dataset, val_dataset, label_to_id = prepare_datasets(df, tokenizer)
    
    os.makedirs(settings.SAVE_PATH, exist_ok=True)
    unique_categories = list(label_to_id.keys())
    with open(settings.labels_file_path, 'w') as f:
        json.dump(unique_categories, f)
    
    train_model(train_dataset, val_dataset, len(unique_categories))
    tokenizer.save_pretrained(settings.SAVE_PATH)
    logger.info(f"Model, tokenizer, and labels successfully saved to {settings.SAVE_PATH}")

if __name__ == '__main__':
    main()