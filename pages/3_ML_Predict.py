import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="วิเคราะห์ ML - 6704062612049", layout="wide", page_icon="🧪")

st.title("🧪 ระบบพยากรณ์ความเสี่ยงสุขภาพ (Ensemble)")
st.write("ส่วนนี้เป็นการนำโมเดลที่ผ่านการเรียนรู้แบบโหวต (Voting) มาทดสอบการใช้งานจริง")

# ส่วนจัดการโมเดล (สร้างอัตโนมัติถ้าไม่มีไฟล์)
if not os.path.exists('models/ensemble_model.pkl'):
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('datasets/heart_data.csv').dropna().drop_duplicates()
    X = df[['Age', 'Cholesterol', 'Stress_Level']]
    y = df['Target']
    m1 = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)), 
        ('lr', LogisticRegression(random_state=42)), 
        ('svc', SVC(probability=True, random_state=42))
    ], voting='soft').fit(X, y)
    pickle.dump(m1, open('models/ensemble_model.pkl', 'wb'))

model = pickle.load(open('models/ensemble_model.pkl', 'rb'))

# ส่วน UI สำหรับกรอกข้อมูล
st.divider()
with st.container(border=True):
    st.subheader("📝 กรุณาระบุข้อมูลเพื่อวิเคราะห์")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("อายุ (ปี)", min_value=1, max_value=100, value=30)
    with col2:
        chol = st.number_input("ระดับคอเลสเตอรอล", min_value=100, max_value=500, value=200)
    stress = st.select_slider("ระดับความเครียดปัจจุบัน (1-10)", options=list(range(1, 11)), value=5)

if st.button("🚀 เริ่มการวิเคราะห์", use_container_width=True):
    with st.spinner('AI กำลังประมวลผล...'):
        pred = model.predict([[age, chol, stress]])
        st.balloons() # แสดงเอฟเฟกต์เมื่อทำงานเสร็จ
        st.subheader("📍 ผลการวิเคราะห์")
        if pred[0] == 1:
            st.error("⚠️ ผลวิเคราะห์: คุณจัดอยู่ในกลุ่มที่มีความเสี่ยงต่อสุขภาพ")
        else:
            st.success("✅ ผลวิเคราะห์: ไม่พบความเสี่ยงที่น่ากังวล สุขภาพของคุณปกติดี")
