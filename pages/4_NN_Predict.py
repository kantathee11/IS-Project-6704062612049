import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="วิเคราะห์รีวิว AI - 6704062612049", layout="wide", page_icon="🧠")

st.title("🧠 ระบบวิเคราะห์ความรู้สึกจากรีวิว (Neural Network)")
st.write("---")

# 2. ส่วนแสดงตารางสถิติ (Statistics Table)
st.subheader("📊 ข้อมูลภาพรวมของ Dataset")
try:
    df_review = pd.read_csv('datasets/review_data.csv')
    # แสดงจำนวน Positive และ Negative แยกกัน
    stats = df_review['Sentiment'].value_counts().rename({1: 'Positive (บวก) 😊', 0: 'Negative (ลบ) 😞'})
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.table(stats)
    with col_t2:
        st.info("💡 ข้อมูลชุดนี้ถูกนำมาฝึกสอน Neural Network เพื่อให้เข้าใจความแตกต่างระหว่างคำชมและคำตำหนิในภาษาอังกฤษ")
except Exception as e:
    st.error("ไม่สามารถโหลดข้อมูลสถิติได้ กรุณาตรวจสอบไฟล์ datasets/review_data.csv")

st.write("---")

# 3. ส่วนเตรียมโมเดล (เน้นความนิ่ง)
@st.cache_resource
def load_and_train_nn():
    df = pd.read_csv('datasets/review_data.csv')
    df['Review_Text'] = df['Review_Text'].str.replace(r'<[^>]*>', '', regex=True).str.lower().str.strip()
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Review_Text'])
    y = df['Sentiment']
    
    # ล็อกค่าสุ่มไว้ที่ 42 เพื่อให้ผลแม่นยำและไม่แกว่ง
    model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
