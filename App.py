import streamlit as st
import os
import requests
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Function to download file from Google Drive
def download_from_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

# URLs to your model files hosted on Google Drive
model_files = {
    "config.json": "1s9Ag8YFisAtcEMc9hXSTw15wLMlcnz6R",
    "merges.txt": "14ETjCKd5rFailuwbxS85B-7BW28BJgYI",
    "model.safetensors": "17cSA1Kd6xqZ27xUDsoK65hXbMMnZsR4p",
    "special_tokens_map.json": "1HZxHafbwhV4fd9p6VgE0jBsrNy0h0Nsj",
    "tokenizer_config.json": "19cF9V6xGlcRy4UIgUhgZsWL67JFgE9Rm",
    "vocab.json": "16XBmWUhoAWGvmX6xYwiRpNZf1REAGSit"
}

model_dir = './pytorch_model'
os.makedirs(model_dir, exist_ok=True)

# Download model files if they do not exist
for filename, file_id in model_files.items():
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        st.info(f"Downloading {filename}...")
        try:
            download_from_drive(file_id, filepath)
            st.info(f"{filename} downloaded.")
        except Exception as e:
            st.error(f"Error downloading {filename}: {e}")
            logging.error(f"Error downloading {filename}: {e}")

# Verify that the model files have been downloaded correctly
for filename in model_files.keys():
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        st.error(f"File {filename} not found in {model_dir}. Please check the file ID and try again.")
        st.stop()
    elif os.path.getsize(filepath) == 0:
        st.error(f"File {filename} is empty. Please check the file ID and try again.")
        st.stop()

# Load the tokenizer and model
try:
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
except json.JSONDecodeError as e:
    logging.error(f"JSON decode error while loading tokenizer: {e}")
    st.error(f"JSON decode error: {e}")
    st.stop()
except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    st.error(f"Failed to load tokenizer: {e}")
    st.stop()

try:
    ai_model = RobertaForSequenceClassification.from_pretrained(model_dir)
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    st.error(f"Failed to load model: {e}")
    st.stop()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ai_model.to(device)

st.title("Text Classification and Toxicity Prediction")

input_text = st.text_area("Enter text to classify")

if st.button("Classify"):
    if input_text:
        # Add your prediction and classification logic here
        st.write("Processing your input...")
    else:
        st.write("Please enter text to classify.")
