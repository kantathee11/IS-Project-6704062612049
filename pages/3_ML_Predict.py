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

# 2. แสดงสถิติข้อมูล
df_health = pd.read_csv('datasets/heart_data.csv').dropna().drop_duplicates()
st.subheader("📊 ตารางสรุปสถิติของชุดข้อมูล (Dataset Statistics)")
st.table(df_health.describe().iloc[[1, 3, 7], :3]) 
st.write("---")

# 3. ส่วนจัดการโมเดล (แก้ปัญหา FileNotFoundError)
model_path = 'models/ensemble_model.pkl'

# ตรวจสอบว่ามีโฟลเดอร์ models หรือยัง ถ้าไม่มีให้สร้าง
if not os.path.exists('models'):
    os.makedirs('models')

# ถ้ายังไม่มีไฟล์โมเดล ให้ทำการฝึกสอนและสร้างไฟล์ใหม่ทันที
if not os.path.exists(model_path):
    with st.status("กำลังสร้างโมเดลครั้งแรก... กรุณารอสักครู่"):
        X = df_health[['Age', 'Cholesterol', 'Stress_Level']]
        y = df_health['Target']
        
        # สร้างโมเดลแบบ Ensemble
        new_model = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)), 
            ('lr', LogisticRegression(random_state=42)), 
            ('svc', SVC(probability=True, random_state=42))
        ], voting='soft').fit(X, y)
        
        # บันทึกไฟล์เก็บไว้ใช้งาน
        with open(model_path, 'wb') as
