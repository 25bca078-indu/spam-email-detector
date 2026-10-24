import os
import streamlit as st
import joblib

# Check if model files exist; if not, run training inline
if not (os.path.exists('spam_model.pkl') and os.path.exists('vectorizer.pkl')):
    import subprocess
    subprocess.run(["python", "train_model.py"])

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("📧 Spam Email Detector")
st.write("Enter an email text or message below to check whether it's **Spam** or **Safe (Ham)**.")

user_input = st.text_area("Paste email content here:", height=150)

if st.button("🔍 Check Email"):
    if user_input.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)[0]
        
        if prediction == 1 or str(prediction).lower() == 'spam':
            st.error("🚨 SPAM DETECTED!")
        else:
            st.success("✅ SAFE / NOT SPAM (HAM)")
            st.info("⚠️ **Reasoning:** This message contains words or patterns commonly associated with scams, phishing, or unsolicited offers.")
        else:
            st.success(f"✅ **SAFE / NOT SPAM (HAM)** (Confidence: {probabilities[0]*100:.2f}%)")
            st.info("ℹ️ **Reasoning:** This message appears to be standard personal or professional communication.")
