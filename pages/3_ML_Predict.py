import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.ensemble import VotingClassifier

st.set_page_config(page_title="วิเคราะห์ ML - 6704062612049", layout="wide", page_icon="🧪")

st.title("🧪 ระบบพยากรณ์ความเสี่ยงสุขภาพ (Ensemble)")
st.write("---")

# โหลดข้อมูลเพื่อแสดงสถิติ
df_health = pd.read_csv('datasets/heart_data.csv').dropna().drop_duplicates()

# ส่วนที่เพิ่ม: ตารางสถิติข้อมูล
st.subheader("📊 ตารางสรุปสถิติของชุดข้อมูล (Dataset Statistics)")
st.write("ข้อมูลภาพรวมของพนักงานทั้งหมดที่ใช้ในการสอน AI:")
# คำนวณค่าสถิติพื้นฐานและแสดงเป็นตาราง
st.table(df_health.describe().iloc[[1, 3, 7], :3]) 
# (หมายเหตุ: iloc เลือกเฉพาะแถว Mean, Min, Max และ 3 คอลัมน์แรก)

st.write("---")

# --- โค้ดส่วนการทำนาย (คงเดิม) ---
if not os.path.exists('models/ensemble_model.pkl'):
    # ... (ส่วนสร้างโมเดลเดิมของคุณ) ...
    pass 

model = pickle.load(open('models/ensemble_model.pkl', 'rb'))

with st.container(border=True):
    st.subheader("📝 กรุณาระบุข้อมูลเพื่อวิเคราะห์")
    c1, c2 = st.columns(2)
    with c1: age = st.number_input("อายุ (ปี):", 1, 100, 30)
    with c2: chol = st.number_input("ระดับคอเลสเตอรอล:", 100, 500, 200)
    stress = st.select_slider("ระดับความเครียด (1-10):", options=list(range(1, 11)), value=5)

    if st.button("🚀 เริ่มการวิเคราะห์", use_container_width=True):
        pred = model.predict([[age, chol, stress]])
        st.subheader("📍 ผลการวิเคราะห์")
        if pred[0] == 1: st.error("⚠️ ผลวิเคราะห์: มีความเสี่ยงต่อสุขภาพ")
        else: 
            st.success("✅ ผลวิเคราะห์: สุขภาพปกติ")
            st.balloons()
