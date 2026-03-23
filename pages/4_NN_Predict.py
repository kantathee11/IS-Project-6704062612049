import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="วิเคราะห์ NN - 6704062612049")
st.title("🧪 วิเคราะห์รีวิวด้วย Neural Network")

# โหลดข้อมูลและคลีนข้อความ
df = pd.read_csv('datasets/review_data.csv')
df['Review_Text'] = df['Review_Text'].str.replace(r'<[^>]*>', '', regex=True).str.lower().str.strip()

# ใช้ Tfidf เพื่อให้น้ำหนักคำที่สำคัญ (เช่น bad, terrible จะถูกให้ค่าความสำคัญสูง)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Review_Text'])
y = df['Sentiment']

# ล็อก random_state เพื่อให้ผลการทำนาย "นิ่ง" ไม่สลับไปมา
model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
model.fit(X, y)

st.write("---")
user_review = st.text_area("พิมพ์รีวิวภาษาอังกฤษที่นี่:", "The food was really bad and it hurt my tongue")

if st.button("ให้ AI วิเคราะห์ความรู้สึก"):
    if user_review:
        # แปลงข้อความเป็นตัวเลข
        input_data = vectorizer.transform([user_review.lower()])
        pred = model.predict(input_data)
        
        # แสดงผลลัพธ์
        if pred[0] == 1:
            st.success("😊 ผลวิเคราะห์: รีวิวเชิงบวก (Positive)")
        else:
            st.error("😞 ผลวิเคราะห์: รีวิวเชิงลบ (Negative)")
    else:
        st.warning("กรุณาพิมพ์ข้อความก่อนกดวิเคราะห์")
