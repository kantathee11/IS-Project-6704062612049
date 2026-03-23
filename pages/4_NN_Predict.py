import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="ทดสอบ NN - 6704062612049")
st.title("🧪 วิเคราะห์รีวิวด้วย Neural Network")

df = pd.read_csv('datasets/review_data.csv')
df['Review_Text'] = df['Review_Text'].str.replace(r'<[^>]*>', '', regex=True).str.lower()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Review_Text'])
y = df['Sentiment']

model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000)
model.fit(X, y)

st.divider()
user_review = st.text_input("ลองพิมพ์รีวิวสินค้า (ภาษาอังกฤษ):", "The food was very good")

if st.button("ให้ AI วิเคราะห์ความรู้สึก"):
    input_data = vectorizer.transform([user_review.lower()])
    pred = model.predict(input_data)
    
    if pred[0] == 1:
        st.success("😊 AI วิเคราะห์ว่า: รีวิวนี้เป็นบวก (Positive)")
    else:
        st.error("😞 AI วิเคราะห์ว่า: รีวิวนี้เป็นลบ (Negative)")
