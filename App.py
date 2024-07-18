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
    try:
        response = requests.post("http://your-flask-api-url/verify_and_check_bias", json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        result = response.json()

        # Display the result
        if "classification" in result and result["classification"] == "AI Generated Text":
            st.markdown(f"### Detected fabrication: **AI generated content**")
            st.markdown(f"**Probability of toxicity:** {result.get('probability_of_toxicity', 0) * 100:.2f}%")
            st.markdown(f"**Prediction:** {'Toxic' if result.get('prediction') else 'Not Toxic'}")
        else:
            st.markdown(f"### **Appears benign**")
            st.markdown(f"**Probability of toxicity:** 0%")
            st.markdown(f"**Prediction:** Not Toxic")

        # Display the metrics if they exist in the result
        st.subheader("Metrics")
        col1, col2 = st.columns(2)
        col1.markdown(f"**AI Score:** {result.get('ai_score', 0) * 100:.2f}%")
        col2.markdown(f"**Toxicity Score:** {result.get('toxicity_score', 0) * 100:.2f}%")
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
    except KeyError as e:
        st.error(f"Missing key in API response: {e}")
