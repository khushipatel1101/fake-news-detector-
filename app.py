import streamlit as st
import joblib
import nltk
import re
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Set up NLTK data path for Streamlit Cloud
nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)

nltk.data.path.append(nltk_data_path)

# ✅ Download required NLTK datasets to /tmp/nltk_data

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)

# Initialize lemmatizer and stopword list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Load trained model
model = joblib.load('fake_news_model.pkl')

# Streamlit app
st.title("📰 Fake News Detector")
st.subheader("Enter the news article text below:")

input_text = st.text_area("News Text", height=200)

if st.button("Predict"):
    if not input_text.strip():
        st.warning("Please enter a news article first.")
    else:
        cleaned = clean_text(input_text)
        prediction = model.predict([cleaned])[0]
        probability = model.predict_proba([cleaned])[0]

        if prediction == 0:
            st.success(f"✅ This is likely REAL news (Confidence: {probability[0]*100:.2f}%)")
        else:
            st.error(f"🚨 This is likely FAKE news (Confidence: {probability[1]*100:.2f}%)")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Random Forest Classifier")
