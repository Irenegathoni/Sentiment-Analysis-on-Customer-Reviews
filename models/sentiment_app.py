import streamlit as st
import pickle
import re
import string

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)  # remove URLs
    text = re.sub(r'\d+', '', text)             # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()    # remove extra spaces
    return text


with open("C:\\Users\\hp\\Downloads\\nlpanalytics\\models\\tfidf_vectorizer.pkl",'rb')as f:
    vectorizer=pickle.load(f)
with open("C:\\Users\\hp\\Downloads\\nlpanalytics\\models\\sentiment_model.pkl",'rb') as f:
    model=pickle.load(f)    

st.title("📢 Sentiment Analysis App") 
st.subheader("Type a sentence and we'll tell you the vibe 😌")  
user_input = st.text_area("Enter your text here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("👀 You tryna test me with empty input? No.")
    else:
        # Clean the input text
        cleaned_input = clean_text(user_input)
        
        # Vectorize the cleaned text
        text_vector = vectorizer.transform([cleaned_input])
        
        # Predict sentiment
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector).max() * 100

        # Display results
        if prediction == "Positive":
            st.success(f"💚 Sentiment: Positive ({probability:.2f}%)")
        else:
            st.error(f"💔 Sentiment: Negative ({probability:.2f}%)")
