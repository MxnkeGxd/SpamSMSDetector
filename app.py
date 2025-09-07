import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Load trained model + vectorizer
# -----------------------------
# (Train your model first, then save with joblib.dump)
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Spam SMS Detector", page_icon="📩")

st.title("📩 Spam SMS Detection App")
st.write("Enter an SMS message below and check if it's **Spam** or **Safe**.")

# Input text box
user_input = st.text_area("✍️ Type your SMS here:", "")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        # Transform input using trained vectorizer
        input_vec = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(input_vec)[0]
        prob = model.predict_proba(input_vec)[0] if hasattr(model, "predict_proba") else None
        
        if prediction == 1:
            st.error("🚨 This looks like **SPAM**!")
        else:
            st.success("✅ This looks like **SAFE** (not spam).")
        
        if prob is not None:
            st.write(f"Confidence: Spam {prob[1]*100:.2f}% | Ham {prob[0]*100:.2f}%")
