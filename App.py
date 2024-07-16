import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib
import requests
from io import BytesIO
import gdown
import os
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Function for basic text preprocessing
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = text.lower()  # Convert to lowercase
    return text

# Download model and tokenizer from Google Drive if not already cached
def download_from_drive(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, output, quiet=True)
        logging.info(f"Downloaded {output} from {url}")

model_files = {
    "config.json": "1s9Ag8YFisAtcEMc9hXSTw15wLMlcnz6R",
    "merges.txt": "14ETjCKd5rFailuwbxS85B-7BW28BJgYI",
    "model.safetensors": "17cSA1Kd6xqZ27xUDsoK65hXbMMnZsR4p",
    "special_tokens_map.json": "1HZxHafbwhV4fd9p6VgE0jBsrNy0h0Nsj",
    "tokenizer_config.json": "19cF9V6xGlcRy4UIgUhgZsWL67JFgE9Rm",
    "vocab.json": "16XBmWUhoAWGvmX6xYwiRpNZf1REAGSit"
}

model_dir = './saved_model'
os.makedirs(model_dir, exist_ok=True)

for filename, file_id in model_files.items():
    download_from_drive(file_id, os.path.join(model_dir, filename))

# Load the AI detection model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained(model_dir)
ai_model = RobertaForSequenceClassification.from_pretrained(model_dir)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ai_model.to(device)

# Load the toxicity prediction model and vectorizer from GitHub
def load_model(url, output_path):
    if not os.path.exists(output_path):
        response = requests.get(url)
        response.raise_for_status()  # Check that the request was successful
        with open(output_path, 'wb') as f:
            f.write(response.content)
        logging.info(f"Downloaded model from {url}")

toxicity_model_path = 'best_xgboost_model.joblib'
vectorizer_path = 'tfidf_vectorizer.joblib'
load_model('https://github.com/Divya-coder-isb/F-B/blob/main/best_xgboost_model.joblib?raw=true', toxicity_model_path)
load_model('https://github.com/Divya-coder-isb/F-B/blob/main/tfidf_vectorizer.joblib?raw=true', vectorizer_path)

toxicity_model = joblib.load(toxicity_model_path)
vectorizer = joblib.load(vectorizer_path)

# Function to predict AI or Human generated text
def predict_ai(text):
    ai_model.eval()
    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = ai_model(**inputs)
        logits = outputs.logits
        score = torch.sigmoid(logits).item()
    return score

# Function to predict toxicity
def predict_toxicity(text, threshold):
    transformed_input = vectorizer.transform([text])
    proba = toxicity_model.predict_proba(transformed_input)[0, 1]
    prediction = (proba >= threshold).astype(int)
    return proba, prediction

# Functions to calculate fairness metrics
def calculate_demographic_parity(predictions, sensitive_attrs):
    rates = predictions.groupby(sensitive_attrs).mean()
    return rates.to_dict()

def calculate_predictive_parity(predictions, labels, sensitive_attrs):
    df = pd.DataFrame({'predictions': predictions, 'labels': labels, 'group': sensitive_attrs})
    positive_pred = df[df['labels'] == 1]
    rates = positive_pred.groupby('group').mean()
    return rates['predictions'].to_dict()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    st.pyplot(plt)

# Streamlit User Interface
st.title("Text Classification and Toxicity Prediction")

input_text = st.text_area("Enter text to classify")

if st.button("Classify"):
    if input_text:
        ai_score = predict_ai(input_text)
        if ai_score > 0.5:
            st.write("AI Generated Text")
            st.write("Running toxicity prediction model...")
            
            # Sidebar for user inputs
            threshold = st.sidebar.slider('Classification Threshold', 0.0, 1.0, 0.237, 0.01)
            
            # Predict toxicity
            proba, prediction = predict_toxicity(input_text, threshold)
            st.write('Probability of Toxicity:', proba)
            st.write('Toxic' if prediction else 'Not Toxic')
            
            # Simulate sensitive attributes and labels for fairness metrics
            sensitive_attrs = np.random.choice(['male', 'female', 'non-binary'], size=100)
            labels = np.random.randint(0, 2, size=100)
            predictions = np.random.rand(100) > 0.5
            
            # Calculate and display fairness metrics
            dp = calculate_demographic_parity(pd.Series(predictions), pd.Series(sensitive_attrs))
            pp = calculate_predictive_parity(pd.Series(predictions), pd.Series(labels), pd.Series(sensitive_attrs))
            st.write("## Fairness Metrics")
            st.write("Demographic Parity:", dp)
            st.write("Predictive Parity:", pp)
            
            # ROC Curve
            plot_roc_curve(labels, predictions)
        else:
            st.write("Human Generated Text")
    else:
        st.write("Please enter text to classify.")
