import streamlit as st
import pickle
import re
import nltk
import string

st.title("SMS spam classifier")

def clean_text(text):
    text = text.lower()  # Lowercase all characters
    text = re.sub(r'@\S+', '', text)  # Remove Twitter handles
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub(r"[^a-zA-Z+']", ' ', text)  # Keep only characters
    text = "".join([i for i in text if i not in string.punctuation])

    words = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')  # Remove stopwords
    text = " ".join([i for i in words if i not in stopwords and len(i) > 2])
    text = re.sub("\s[\s]+", " ", text).strip()  # Remove repeated/leading/trailing spaces
    return text

text = st.text_area(label='Enter SMS message')
text = clean_text(text)

if st.button(label='Predict'):
    tfidf = pickle.load(open('tfidf.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))

    vector_text = tfidf.transform([text])
    sol = model.predict(vector_text.toarray())
    if sol == 1:
        st.header('Spam')
    else:
        st.header('Not spam')