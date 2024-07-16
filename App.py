import streamlit as st
import os
import requests
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Function to download file from a URL
def download_file(url, destination):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    logging.info(f"Downloaded {destination} from {url}")

# URLs to your model files hosted on cloud storage
model_files = {
    "config.json": "https://your-cloud-storage-link/config.json",
    "merges.txt": "https://your-cloud-storage-link/merges.txt",
    "pytorch_model.bin": "https://your-cloud-storage-link/pytorch_model.bin",
    "special_tokens_map.json": "https://your-cloud-storage-link/special_tokens_map.json",
    "tokenizer_config.json": "https://your-cloud-storage-link/tokenizer_config.json",
    "vocab.json": "https://your-cloud-storage-link/vocab.json"
}

model_dir = './pytorch_model'
os.makedirs(model_dir, exist_ok=True)

# Download model files if they do not exist
for filename, url in model_files.items():
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        st.info(f"Downloading {filename}...")
        download_file(url, filepath)
        st.info(f"{filename} downloaded.")

# Verify that the model files have been downloaded correctly
for filename in model_files.keys():
    if not os.path.exists(os.path.join(model_dir, filename)):
        st.error(f"File {filename} not found in {model_dir}. Please check the file URL and try again.")
        st.stop()

# Load the tokenizer and model
try:
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
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
