import streamlit as st
import requests
import matplotlib.pyplot as plt
from PIL import Image

# Function to classify text and predict toxicity
def classify_text(text, threshold, url):
    input_data = {"text": text, "ai_score_threshold": threshold}
    response = requests.post(url, json=input_data)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Define the ngrok URL
url =  "https://91be-34-125-248-172.ngrok-free.app/verify_and_check_bias"

# Load the banner image from the URL
banner_url = "https://github.com/Griffender/Fake-berry/raw/main/Banner.png"
banner_image = Image.open(requests.get(banner_url, stream=True).raw)

# Display the banner image as a header
st.image(banner_image, use_column_width=True)

# Streamlit app
st.title("AI vs Human Text Classification")
st.write("Enter a text to test whether it is generated by AI or a human.")

# Create columns for layout
left_col, right_col = st.columns(2)

# Input text box in the left column
input_text = left_col.text_area("Input Text", height=200)
ai_score_threshold = left_col.slider("AI Score Threshold", 0.0, 1.0, 0.5)

if left_col.button("Classify"):
    if input_text:
        result = classify_text(input_text, ai_score_threshold, url)
        if result:
            classification = result.get("classification", "N/A")
            left_col.write("**Classification Result:**")
            left_col.write(f"Classification: {classification}")

            if classification == "AI Generated Text":
                toxicity_result = classify_text(input_text, ai_score_threshold, url)
                if toxicity_result:
                    probability_of_toxicity = toxicity_result.get("probability_of_toxicity", 0.0)
                    prediction = toxicity_result.get("prediction", "N/A")

                    left_col.write(f"Prediction: {prediction}")

                    # Plot the circular progress chart for toxicity score
                    fig, ax = plt.subplots()
                    ax.pie([probability_of_toxicity, 1 - probability_of_toxicity], 
                           startangle=90, colors=['#FF6F61', '#E0E0E0'], 
                           wedgeprops={'width': 0.3})
                    ax.text(0, 0, f"{int(probability_of_toxicity * 100)}%", 
                            ha='center', va='center', fontsize=20, color='#FF6F61')
                    ax.set_aspect('equal')

                    # Display the chart in the right column
                    with right_col:
                        st.pyplot(fig)
                else:
                    left_col.error("Error: Unable to classify the text for toxicity. Please try again later.")
        else:
            left_col.error("Error: Unable to classify the text. Please try again later.")
    else:
        left_col.warning("Please enter some text to classify.")
