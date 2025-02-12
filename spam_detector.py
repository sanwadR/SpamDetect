import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
import urllib.request

# Download the model from GitHub
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/sanwadR/SpamDetect/main/spam_classifier.pkl",
    "spam_classifier.pkl"
)

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/sanwadR/SpamDetect/main/tfidf_vectorizer.pkl",
    "tfidf_vectorizer.pkl"
)

# Load the model
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("Spam Detector")
st.write("Enter a message to check whether it's spam or not.")

# User input
user_input = st.text_area("Enter your message here:", "")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform input text
        input_vectorized = vectorizer.transform([user_input])

        # Predict and get probabilities
        prediction = model.predict(input_vectorized)[0]
        probabilities = model.predict_proba(input_vectorized)

        st.write(f"Prediction Probabilities: {probabilities}")

        if prediction == 0:
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **HAM** (Not Spam).")
