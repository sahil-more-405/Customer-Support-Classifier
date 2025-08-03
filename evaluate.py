import warnings
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from predict import classify_queries, LABELS_FILE
import os
import json

def evaluate_hardcoded_model():
    print("--- Starting Hardcoded Model Evaluation with Easier Queries ---")

    data = {
        'id': range(1, 21),
        'customer_id': range(101, 121),
        'query': [
            "Hey, I need to send something back. What's the process?",
            "The tracking info hasn't updated in days. Is my shipment lost?",
            "Is there any coverage if this product stops working later?",
            "Can I call off my purchase? I changed my mind.",
            "The item came in damaged. How do I send it back?",
            "Still no sign of my delivery. Can you help?",
            "I want to know how the warranty works for this thing.",
            "Is there a way to stop my subscription from renewing?",
            "My order says it's delivered but I didn't get it.",
            "I've returned something but haven’t received my money.",
            "My gadget just died—what does the warranty say about this?",
            "Can I stop just one product from shipping in my order?",
            "Any update on my parcel? It's been days.",
            "You sent me the wrong product!",
            "Don’t need the item anymore. Want to return.",
            "Does this kind of issue count under warranty?",
            "Need help fixing the address I gave for shipping.",
            "How do returns work for clearance stuff?",
            "I’m done—want out of this account.",
            "It's faulty. Can I swap it under the warranty terms?"
        ],
        'category': [
            "return",
            "shipping",
            "warranty",
            "cancellation",
            "return",
            "shipping",
            "warranty",
            "cancellation",
            "shipping",
            "return",
            "warranty",
            "cancellation",
            "shipping",
            "return",
            "return",
            "warranty",
            "shipping",
            "return",
            "cancellation",
            "warranty"
        ]

    }

    df = pd.DataFrame(data)

    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"Labels file not found at {LABELS_FILE}. Please run train_model.py first.")
    
    with open(LABELS_FILE, 'r') as f:
        LABELS = json.load(f)

    queries_for_prediction = [
        (row['id'], row['customer_id'], row['query'])
        for index, row in df.iterrows()
    ]
    
    y_true = df['category'].tolist()
    print(f"Loaded {len(y_true)} hardcoded entries for evaluation.")

    classified_results = classify_queries(queries_for_prediction)
    y_pred = [res[3] for res in classified_results]

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    print("\nCorrectly Classified Samples:")
    correct_predictions = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct_predictions.append({'query': df.iloc[i]['query'], 'true': y_true[i], 'pred': y_pred[i]})
    
    for pred in correct_predictions[:3]:
        print(f"- Query: '{pred['query']}' | Predicted: {pred['pred']}")

    print("\nIncorrectly Classified Samples:")
    incorrect_predictions = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            incorrect_predictions.append({'query': df.iloc[i]['query'], 'true': y_true[i], 'pred': y_pred[i]})
    
    for pred in incorrect_predictions[:3]:
        print(f"- Query: '{pred['query']}' | True: {pred['true']} | Predicted: {pred['pred']}")

    
    print("\n--- Evaluation Complete ---")
    
if __name__ == "__main__":
    evaluate_hardcoded_model()