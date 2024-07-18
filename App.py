import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Set up the page configuration
st.set_page_config(page_title="Fake Berry", page_icon=":robot_face:", layout="wide")

# Load the Fake Berry banner image
banner_image_url = "https://github.com/Griffender/Fake-berry/blob/main/Banner.png?raw=true"
response = requests.get(banner_image_url)
banner_image = Image.open(BytesIO(response.content))

# Display the banner
st.image(banner_image, use_column_width=True)

# Title
st.title("Fake Berry: Your Ethical Watchdog")

# Query input
st.subheader("Craft Your Query: Unmask Bias, Confirm Fairness, and Validate Authenticity")
user_query = st.text_area("Enter text here:")

# Tailor your metrics
st.subheader("Tailor your metrics")
ai_score_threshold = st.slider("AI Score Threshold", 0.0, 1.0, 0.45)
toxicity_threshold = st.slider("Toxicity Threshold", 0.0, 1.0, 0.6)

# Submit button
if st.button("Apply"):
    # Send the user query to the Flask API
    payload = {
        "text": user_query,
        "threshold": toxicity_threshold,
        "ai_score_threshold": ai_score_threshold
    }
    response = requests.post("https://d0b3-34-16-187-100.ngrok-free.app/verify_and_check_bias", json=payload)
    result = response.json()

    # Display the result
    if result["classification"] == "AI Generated Text":
        st.markdown(f"### Detected fabrication: **AI generated content**")
        st.markdown(f"**Probability of toxicity:** {result['probability_of_toxicity']*100:.2f}%")
        st.markdown(f"**Prediction:** {'Toxic' if result['prediction'] else 'Not Toxic'}")
    else:
        st.markdown(f"### **Appears benign**")
        st.markdown(f"**Probability of toxicity:** 0%")
        st.markdown(f"**Prediction:** Not Toxic")

    # Display the metrics
    st.subheader("Metrics")
    col1, col2 = st.columns(2)
    col1.markdown(f"**AI Score:** {result['ai_score']*100:.2f}%")
    col2.markdown(f"**Toxicity Score:** {result['toxicity_score']*100:.2f}%")

