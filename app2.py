#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:
import nltk 
nltk.download('punkt')

import streamlit as st
import pandas as pd
import numpy as np
import os
import pysrt
import nltk
import tempfile
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.svm import SVC
import joblib
# Загрузка модели
model = joblib.load('./model9.pkl')



vectorizer = joblib.load('./vectorizer2.pkl')



# Функция для предсказания уровня сложности фильма на основе загруженных субтитров
def predict_level(subtitles):
    stemmer = nltk.stem.PorterStemmer()
    words = nltk.word_tokenize(subtitles.lower())
    stemmed_words = [stemmer.stem(word) for word in words]
    subtitles_stemmed = ' '.join(stemmed_words)
    X = vectorizer.transform([subtitles_stemmed])
    y_pred = model.predict(X)
    return y_pred[0]

# Создание интерфейса приложения
st.title("Определение уровня сложности фильма по субтитрам")
st.write("Загрузите файл с субтитрами и узнайте уровень сложности фильма")
uploaded_file = st.file_uploader("Выберите файл", type=['srt'])
if uploaded_file is not None and uploaded_file.name.endswith('.srt'):
    # Сохраняем содержимое файла во временный файл
    with tempfile.NamedTemporaryFile(delete=False) as tempfile:
        tempfile.write(uploaded_file.read())
        abs_path = os.path.abspath(tempfile.name)
    # Читаем субтитры из временного файла
    subs = pysrt.open(abs_path, encoding='windows-1252')
    text = ""
    for sub in subs:
        text += sub.text + " "
    level = predict_level(text)
    st.write("Уровень сложности фильма:", level)
else:
    st.write("Загрузите файл с субтитрами формата .srt")


# In[ ]:




