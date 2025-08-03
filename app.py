import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import os
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# --- Configuration ---
# It's good practice to manage settings via a class or environment variables
# rather than hardcoding them.
class Settings:
    MODEL_PATH = "./model/fine_tuned_model"
    LABELS_FILE_NAME = "labels.json"
    HOST = "127.0.0.1"
    PORT = 8000
    LOG_LEVEL = "INFO"

    @property
    def labels_file_path(self) -> str:
        return os.path.join(self.MODEL_PATH, self.LABELS_FILE_NAME)

settings = Settings()

# --- Logging Setup ---
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# --- Pre-load HTML content ---
# Load HTML content once at startup to avoid reading the file on every request.
try:
    with open("index.html") as f:
        HTML_CONTENT = f.read()
except FileNotFoundError:
    logger.error("index.html not found. UI will not be available.")
    HTML_CONTENT = "<h1>UI file not found</h1>"

# --- FastAPI Lifespan for Model Loading/Unloading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events. Loads the model on startup
    and cleans up on shutdown.
    """
    logger.info("Loading model and tokenizer...")
    if not os.path.exists(settings.labels_file_path):
        raise FileNotFoundError(f"Labels file not found at {settings.labels_file_path}. Please run train_model.py first.")

    # Store ML assets in the app state
    app.state.ml_models = {}

    # Determine device and load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.ml_models["device"] = device
    logger.info(f"Using device: {device}")

    # Load labels
    with open(settings.labels_file_path, 'r') as f:
        labels = json.load(f)
    app.state.ml_models["id_to_label"] = {i: label for i, label in enumerate(labels)}

    # Load tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained(settings.MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(settings.MODEL_PATH)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    app.state.ml_models["tokenizer"] = tokenizer
    app.state.ml_models["model"] = model
    
    logger.info("Model and tokenizer loaded successfully.")
    
    yield
    
    # Clean up the ML models and release the resources
    logger.info("Cleaning up ML models.")
    app.state.ml_models.clear()

# Initialize FastAPI app with the lifespan event handler
app = FastAPI(lifespan=lifespan)

# --- Pydantic Models for API ---
class Query(BaseModel):
    id: int
    customer_id: int
    text: str

class ClassifiedQuery(BaseModel):
    id: int
    customer_id: int
    text: str
    category: str

# --- Prediction Logic ---
def classify_queries(
    queries: List[Query], 
    model: BertForSequenceClassification, 
    tokenizer: BertTokenizerFast, 
    device: str, 
    id_to_label: Dict[int, str]
) -> List[ClassifiedQuery]:
    """
    Classifies a batch of queries using the provided model and tokenizer.
    """
    if not queries:
        return []

    query_texts = [q.text for q in queries]

    # Tokenize and move to the correct device
    inputs = tokenizer(
        query_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=1)
    
    # Combine queries with their predicted categories using a list comprehension
    return [
        ClassifiedQuery(
            id=query.id,
            customer_id=query.customer_id,
            text=query.text,
            category=id_to_label[prediction.item()]
        )
        for query, prediction in zip(queries, predictions)
    ]

# --- API Endpoint ---
@app.get("/")
def read_root():
    """
    Root endpoint to provide a welcome message and API status.
    """
    return {"message": "Text Classifier API is running. Send a POST request to /predict/ to classify text."}

@app.get("/ui", response_class=HTMLResponse)
async def read_ui():
    """
    Serves the user interface HTML page.
    """
    return HTMLResponse(content=HTML_CONTENT)

@app.post("/predict/", response_model=List[ClassifiedQuery])
def predict(queries: List[Query], request: Request):
    """
    Receives a list of queries, classifies them, and returns the results.
    """
    try:
        logger.info(f"Received {len(queries)} queries for prediction.")
        results = classify_queries(
            queries=queries,
            model=request.app.state.ml_models["model"],
            tokenizer=request.app.state.ml_models["tokenizer"],
            device=request.app.state.ml_models["device"],
            id_to_label=request.app.state.ml_models["id_to_label"]
        )
        logger.info(f"Successfully classified {len(results)} queries.")
        return results
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during model prediction.")

# --- Main Block to Run the App ---
if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        log_level=settings.LOG_LEVEL.lower(),
        reload=True # Use reload for development; disable in production
    )
