import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="วิเคราะห์ ML - 6704062612049", layout="wide", page_icon="🧪")

st.title("🧪 ระบบพยากรณ์ความเสี่ยงสุขภาพ (Ensemble)")
st.write("---")

# 2. ส่วนแสดงตารางสถิติ (Statistics Table)
st.subheader("📊 ตารางสรุปสถิติของชุดข้อมูล (Dataset Statistics)")
try:
    df_health = pd.read_csv('datasets/heart_data.csv').dropna().drop_duplicates()
    # แสดงค่า Mean, Min, Max ของตัวแปรหลัก
    st.table(df_health[['Age', 'Cholesterol', 'Stress_Level']].describe().loc[['mean', 'min', 'max']])
except Exception as e:
    st.error("ไม่สามารถโหลดตารางสถิติได้ กรุณาตรวจสอบไฟล์ datasets/heart_data.csv")

st.write("---")

# 3. ระบบจัดการโมเดล (ตรวจสอบและสร้างไฟล์อัตโนมัติ)
model_path = 'models/ensemble_model.pkl'
if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists(model_path):
    with st.status("กำลังเตรียมระบบวิเคราะห์..."):
        X = df_health[['Age', 'Cholesterol', 'Stress_Level']]
        y = df_health['Target']
        # ฝึกสอนโมเดลใหม่
        m = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)), 
            ('lr', LogisticRegression(random_state=42)), 
            ('svc', SVC(probability=True, random_state=42))
        ], voting='soft').fit(X, y)
        with open(model_path, 'wb') as f:
            pickle.dump(m, f)
        st.success("ระบบพร้อมใช้งาน!")

# โหลดโมเดล
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 4. ส่วนรับข้อมูลจากผู้ใช้
with st.container(border=True):
    st.subheader("📝 กรุณาระบุข้อมูลเพื่อวิเคราะห์")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("อายุ (ปี):", 1, 100, 30)
    with c2:
        chol = st.number_input("ระดับคอเลสเตอรอล:", 100, 500, 200)
    stress = st.select_slider("ระดับความเครียด (1-10):", options=list(range(1, 11)), value=5)

    if st.button("🚀 เริ่มการวิเคราะห์", use_container_width=True):
        pred = model.predict([[age, chol, stress]])
        st.write("---")
        st.subheader("📍 ผลการวิเคราะห์")
        if pred[0] == 1:
            st.error("⚠️ ผลวิเคราะห์: คุณจัดอยู่ในกลุ่มที่มีความเสี่ยงต่อสุขภาพ")
        else:
            st.success("✅ ผลวิเคราะห์: สุขภาพของคุณปกติดี")
            st.balloons()

st.write("---")
st.info("💡 หมายเหตุ: โมเดลนี้พัฒนาขึ้นโดยใช้เทคนิค Ensemble Learning เพื่อความแม่นยำในข้อมูลจำลอง")
