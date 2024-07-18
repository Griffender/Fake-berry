import streamlit as st
import requests
import matplotlib.pyplot as plt
from PIL import Image

# Define the ngrok URL
url = "https://eab2-34-16-216-78.ngrok-free.app/verify_and_check_bias"

# Load the banner image from the URL
banner_url = "https://github.com/Griffender/Fake-berry/raw/main/Banner.png"
banner_image = Image.open(requests.get(banner_url, stream=True).raw)

# Display the banner image
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
        # Define the input data for AI classification
        input_data = {
            "text": input_text,
            "ai_score_threshold": ai_score_threshold
        }

        # Send a POST request to the endpoint
        response = requests.post(url, json=input_data)

        # Check the response status
        if response.status_code == 200:
            result = response.json()
            # Display the result as normal text
            classification = result.get("classification", "N/A")
            left_col.write("**Classification Result:**")
            left_col.write(f"Classification: {classification}")

            if classification == "AI Generated Text":
                # Define the input data for toxicity prediction
                response_toxicity = requests.post(url, json=input_data)

                # Check the response status
                if response_toxicity.status_code == 200:
                    result_toxicity = response_toxicity.json()
                    probability_of_toxicity = result_toxicity.get("probability_of_toxicity", 0.0)
                    prediction = result_toxicity.get("prediction", "N/A")

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
