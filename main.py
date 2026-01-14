from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words("english"))
lemma = WordNetLemmatizer()

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("TFIDF.pkl", "rb") as file:
    tfidf = pickle.load(file)

class Data(BaseModel):
    text_input: str

app = FastAPI(title="Email Spam Classification API")

def text_preprocess(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text, preserve_line=True)
    words = [
        lemma.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(words)

@app.post("/predict")
def predict(data: Data):
    processed_text = text_preprocess(data.text_input)
    vector = tfidf.transform([processed_text])
    output = model.predict(vector)[0]

    if output == 1:
        return {"prediction": 1, "label": "Mail is SPAM"}
    else:
        return {"prediction": 0, "label": "Mail is NOT SPAM"}
