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

# Convert the banner image to base64
import base64
from io import BytesIO

buffered = BytesIO()
banner_image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

# Display the banner with adjusted size
banner_html = f"""
<div style="width: 100%; overflow: hidden; background-color: #f1f1f1; text-align: center;">
    <img src="data:image/png;base64,{img_str}" style="width: 2000px; height: 150px;" />
</div>
"""

st.markdown(banner_html, unsafe_allow_html=True)

# Title
st.title("Fake Berry: Your Ethical Watchdog")

# Query input
st.subheader("Craft Your Query: Unmask Bias, Confirm Fairness, and Validate Authenticity")
user_query = st.text_area("Enter text here:")

# Tailor your metrics
st.subheader("Tailor your metrics")
ai_score_threshold = st.slider("AI Score Threshold", 0.0, 1.0, 0.45)

# Variables to hold toxicity result and updated result
toxicity_result = None

# Submit button
if st.button("Apply"):
    # Send the user query to the Flask API
    payload = {
        "text": user_query,
        "ai_score_threshold": ai_score_threshold
    }
    try:
        response = requests.post("https://eab2-34-16-216-78.ngrok-free.app/verify_and_check_bias", json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        result = response.json()

        # Display the result
        if "classification" in result and result["classification"] == "AI Generated Text":
            st.markdown(f"### Detected fabrication: **AI generated content**")
            st.markdown(f"**Probability of toxicity:** {result.get('probability_of_toxicity', 0) * 100:.2f}%")
            st.markdown(f"**Prediction:** {'Toxic' if result.get('prediction') else 'Not Toxic'}")
            toxicity_threshold = st.slider("Toxicity Threshold", 0.0, 1.0, 0.6, key="toxicity_threshold")

            # Re-check the toxicity with the new threshold if needed
            if st.button("Re-check Toxicity"):
                payload["threshold"] = toxicity_threshold
                payload["text"] = user_query  # Include the text again in the payload
                payload["ai_score_threshold"] = ai_score_threshold  # Include the AI score threshold in the payload
                response = requests.post("http://your-flask-api-url/verify_and_check_bias", json=payload)
                response.raise_for_status()
                updated_result = response.json()
                st.markdown(f"**Updated Probability of toxicity:** {updated_result.get('probability_of_toxicity', 0) * 100:.2f}%")
                st.markdown(f"**Updated Prediction:** {'Toxic' if updated_result.get('prediction') else 'Not Toxic'}")
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
