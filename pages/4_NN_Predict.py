import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# ตั้งค่าหน้าเว็บให้เข้ากับหน้าอื่นๆ
st.set_page_config(page_title="วิเคราะห์รีวิว AI - 6704062612049", layout="wide", page_icon="🧠")

st.title("🧠 ระบบวิเคราะห์ความรู้สึกจากรีวิว (Neural Network)")
st.write("ใช้โครงข่ายประสาทเทียมในการจำแนกข้อความว่าเป็นเชิงบวกหรือเชิงลบ")
st.write("---")

# ส่วนเตรียมโมเดล (เน้นความนิ่งของผลลัพธ์)
@st.cache_resource
def load_and_train():
    # โหลด Dataset รีวิวที่เราเตรียมไว้
    df = pd.read_csv('datasets/review_data.csv')
    # ทำความสะอาดข้อมูลเบื้องต้น (ลบช่องว่างและปรับเป็นตัวพิมพ์เล็ก)
    df['Review_Text'] = df['Review_Text'].str.replace(r'<[^>]*>', '', regex=True).str.lower().str.strip()
    
    # แปลงข้อความเป็นตัวเลขด้วยเทคนิค Tfidf
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Review_Text'])
    y = df['Sentiment']
    
    # สร้างโมเดล Neural Network (ล็อกค่าสุ่มไว้ที่ 42 เพื่อให้ผลลัพธ์คงที่)
    model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = load_and_train()

# ส่วนแสดงการใช้งาน
with st.container(border=True):
    st.subheader("📝 พิมพ์ข้อความรีวิว")
    st.write("กรุณาป้อนข้อความภาษาอังกฤษเพื่อตรวจสอบความรู้สึกที่ซ่อนอยู่")
    
    # ช่องรับข้อมูลข้อความ
    user_text = st.text_input("ระบุข้อความรีวิว:", "The experience was very good!")
    
    if st.button("🚀 วิเคราะห์ความรู้สึก", use_container_width=True):
        if user_text:
            # ประมวลผลและทำนาย
            input_vector = vectorizer.transform([user_text.lower()])
            prediction = model.predict(input_vector)
            
            st.write("---")
            st.subheader("📍 ผลการวิเคราะห์")
            
            if prediction[0] == 1:
                st.success("😊 AI วิเคราะห์ว่าเป็น: **รีวิวเชิงบวก (Positive)**")
                st.balloons()
            else:
                st.error("😞 AI วิเคราะห์ว่าเป็น: **รีวิวเชิงลบ (Negative)**")
        else:
            st.warning("กรุณากรอกข้อความก่อนกดปุ่มวิเคราะห์")

# ส่วนท้ายหน้า
st.write("---")
st.info("💡 หมายเหตุ: โมเดลนี้ถูกฝึกสอนด้วยข้อมูลรีวิวจำลอง (Synthetic Data) เพื่อการศึกษาในรายวิชา IS")
