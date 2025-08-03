import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import os
import json
from typing import List, Tuple

# --- Constants ---
MODEL_PATH = "./model/fine_tuned_model"
LABELS_FILE = os.path.join(MODEL_PATH, "labels.json")

# --- Load Model and Tokenizer ---
# Determine the device to run the model on
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the fine-tuned model and tokenizer
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}. Please run train_model.py first.")

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()  # Set the model to evaluation mode

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

# Load the labels
if not os.path.exists(LABELS_FILE):
    raise FileNotFoundError(f"Labels file not found at {LABELS_FILE}. Please run train_model.py first.")

with open(LABELS_FILE, 'r') as f:
    LABELS = json.load(f)

# Create a mapping from index to label
id_to_label = {i: label for i, label in enumerate(LABELS)}

# --- Prediction Function ---
def classify_queries(queries: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str, str]]:
    """
    Classifies a list of queries using the fine-tuned model.

    Args:
        queries: A list of tuples, where each tuple contains (id, customer_id, query_text).

    Returns:
        A list of tuples, where each tuple contains (id, customer_id, query_text, predicted_category).
    """
    if not queries:
        return []

    # Extract the query texts from the input tuples
    query_texts = [q[2] for q in queries]

    # Tokenize the input texts
    inputs = tokenizer(query_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move tensors to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class indices
    predictions = torch.argmax(outputs.logits, dim=1)

    # Map predictions to labels and format the output
    results = []
    for i, query in enumerate(queries):
        pred_index = predictions[i].item()
        predicted_category = id_to_label[pred_index]
        results.append((query[0], query[1], query[2], predicted_category))
    
    return results

# --- Example Usage ---
if __name__ == '__main__':
    # Example queries to classify
    example_queries = [
        (1, 101, "I want to return my item, it's broken."),
        (2, 102, "What is the warranty on this product?"),
        (3, 103, "I need to cancel my order."),
        (4, 104, "Where is my package? It hasn't arrived yet.")
    ]

    # Classify the queries
    classified_results = classify_queries(example_queries)

    # Print the results
    for result in classified_results:
        print(f"ID: {result[0]}, Customer ID: {result[1]}, Query: '{result[2]}', Category: {result[3]}")
