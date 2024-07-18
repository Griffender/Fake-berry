import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
 
# Set up the page configuration
st.set_page_config(page_title="Fake Berry", page_icon=":robot_face:", layout="wide")
 
# Load the Fake Berry banner image
banner_image_url = "https://github.com/Griffender/Fake-berry/blob/main/Banner.png?raw=true"
response = requests.get(banner_image_url)
banner_image = Image.open(BytesIO(response.content))
 
# Convert the banner image to base64
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
 
# Layout for input and metrics
with st.container():
    col1, col2 = st.columns([2, 1])
 
    with col1:
        st.subheader("Craft Your Query: Unmask Bias, Confirm Fairness, and Validate Authenticity")
        user_query = st.text_area("Enter text here:", height=200)
 
    with col2:
        st.markdown(f"<div style='text-align: center;'><span style='color: red; font-weight: bold;'>Detected fabrication: AI generated content</span></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center;'>Probability of toxicity:</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; font-weight: bold;'>87%</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center;'>Prediction: <span style='color: green; font-weight: bold;'>Appears benign</span></div>", unsafe_allow_html=True)
        st.image("https://github.com/Griffender/Fake-berry/blob/main/Banner.png", width=150)
 
# Tailor your metrics
st.subheader("Tailor your metrics")
ai_score_threshold = st.slider("AI Score Threshold", 0.0, 1.0, 0.45)
toxicity_threshold = st.slider("Toxicity Threshold", 0.0, 1.0, 0.6)
 
# Submit button
if st.button("Apply"):
    # Send the user query to the Flask API
    payload = {
        "text": user_query,
        "ai_score_threshold": ai_score_threshold,
        "threshold": toxicity_threshold
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
