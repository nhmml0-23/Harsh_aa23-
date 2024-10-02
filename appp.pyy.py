#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[16]:


# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="streamlit")

# Load the saved model and vectorizer
def load_model_and_vectorizer():
    try:
        with open('model_DT.pkl', 'rb') as f:
            model_DT = pickle.load(f)
        with open('tfidf_vector.pkl', 'rb') as f:
            tfidf_vector = pickle.load(f)
        return model_DT, tfidf_vector
    except (FileNotFoundError, EOFError) as e:
        st.error(f"Error loading model or vectorizer: {str(e)}")
        return None

loaded_models = load_model_and_vectorizer()

if loaded_models is not None:
    model_DT, tfidf_vector = loaded_models

    # Create a Streamlit app
    st.title("Resume Classification App")
    st.write("This app classifies resumes into different categories.")

    # Upload resume file
    st.subheader("Upload Resume")
    resume_file = st.file_uploader("Choose a file", type=["pdf", "docx", "doc"])

    # Extract text from resume
    def extract_text(resume_file):
        if resume_file.endswith('.pdf'):
            import PyPDF2
            pdf_file = PyPDF2.PdfFileReader(resume_file)
            text = ''
            for page in range(pdf_file.numPages):
                text += pdf_file.getPage(page).extractText()
        elif resume_file.endswith('.docx'):
            import docx2txt
            text = docx2txt.process(resume_file)
        elif resume_file.endswith('.doc'):
            import docx2txt
            text = docx2txt.process(resume_file)
        return text

    # Preprocess text
    def preprocess_text(text):
        import nltk
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    # Classify resume
    def classify_resume(resume_text):
        resume_text = preprocess_text(resume_text)
        resume_text = tfidf_vector.transform([resume_text])
        prediction = model_DT.predict(resume_text)
        return prediction

    # Run the app
    if resume_file:
        resume_text = extract_text(resume_file)
        prediction = classify_resume(resume_text)
        st.write("Predicted Category:", prediction)





# In[ ]:




