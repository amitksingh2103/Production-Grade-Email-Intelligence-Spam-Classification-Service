import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('TFIDF.pkl', 'rb') as f:
    tfidf = pickle.load(f)

st.set_page_config(
    page_title="Email Intelligence & Spam Classification",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Intelligence & Spam Classification Service")
st.write("Analyze email messages using a precision-optimized classical ML pipeline.")


text = st.text_area(
    "Enter the email content below",
    height=200,
    placeholder="Paste the email message here..."
)

def text_preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    words = [lemma.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(words)

if st.button('Predict'):
    processed_text = text_preprocess(text)
    vector = tfidf.transform([processed_text])
    output = model.predict(vector)[0]

    if output == 1:
        st.write('Mail is SPAM')
    else:
        st.write('Mail is Not SPAM')
