import streamlit as st
import joblib

# Set up webpage title and layout
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")

# Webpage Header
st.title("📧 Spam Email Detector")
st.write("Enter an email text or message below to check whether it's **Spam** or **Safe (Ham)**.")

# Load the trained model and vectorizer saved in Step 3
@st.cache_resource
def load_assets():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except Exception as e:
    st.error("Model files not found! Please run `python train_model.py` first.")
    st.stop()

# Text input box for the email content
user_input = st.text_area("✉️ Email Content / Message:", height=150, placeholder="Paste email body here...")

# Button to trigger the AI prediction
if st.button("🔍 Check Email"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # 1. Convert user input text into numbers
        vectorized_text = vectorizer.transform([user_input])
        
        # 2. Predict using our trained AI model
        prediction = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]
        
        # 3. Display the result on screen
        st.markdown("---")
        if prediction == 1:
            st.error(f"🚨 **SPAM DETECTED!** (Confidence: {probabilities[1]*100:.2f}%)")
            st.info("⚠️ **Reasoning:** This message contains words or patterns commonly associated with scams, phishing, or unsolicited offers.")
        else:
            st.success(f"✅ **SAFE / NOT SPAM (HAM)** (Confidence: {probabilities[0]*100:.2f}%)")
            st.info("ℹ️ **Reasoning:** This message appears to be standard personal or professional communication.")
