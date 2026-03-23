import streamlit as st
import pandas as pd

# 1. ตั้งค่าหน้าเว็บให้ดูเป็นมืออาชีพ
st.set_page_config(page_title="ML Theory - 6704062612049", layout="wide", page_icon="📚")

st.title("📚 รายละเอียดการพัฒนาโมเดล Machine Learning")
st.write("---")

# ส่วนแสดงสถิติแบบ Metric ให้ดูน่าสนใจ
m1, m2, m3 = st.columns(3)
m1.metric("Dataset Name", "heart_data.csv")
m2.metric("Model Type", "Ensemble (Voting)")
m3.metric("Status", "Data Cleaned")

# a. ระบุที่มาของ Dataset
with st.expander("📌 A. ที่มาของข้อมูล (Dataset Source)", expanded=True):
    st.write("ข้อมูลชุดนี้เป็น **ข้อมูลจำลอง (Synthetic Data)** ที่สร้างขึ้นโดย Generative AI เพื่อจำลองสถานการณ์การวิเคราะห์ความเสี่ยงด้านสุขภาพของพนักงานในองค์กร สำหรับใช้ในโปรเจควิชา IS 2568")

# b. อธิบาย feature ของ Dataset
with st.expander("📊 B. คำอธิบายตัวแปร (Features)", expanded=True):
    st.write("ข้อมูลชุด 'heart_data.csv' ประกอบด้วยตัวแปรหลักที่ใช้ในการพยากรณ์ ดังนี้:")
    st.markdown("""
    * **Age:** อายุของพนักงาน (ตัวเลข)
    * **Cholesterol:** ระดับไขมันในเลือด
    * **Stress_Level:** ระดับความเครียดประเมินจากแบบสอบถาม (คะแนน 1-10)
    * **Target:** ผลลัพธ์การทำนาย (0 = ปกติ, 1 = เสี่ยงโรค)
    """)

# c. ความไม่สมบูรณ์และการเตรียมข้อมูล
with st.expander("🛠️ C. การจัดการข้อมูลที่ไม่สมบูรณ์ (Data Preparation)", expanded=True):
    st.warning("⚠️ ตรวจพบความไม่สมบูรณ์: มีค่าว่าง (Missing Values) และข้อมูลซ้ำ (Duplicates)")
    st.write("**ขั้นตอนการเตรียมข้อมูล:**")
    st.markdown("""
    1. **การลบค่าว่าง:** ใช้คำสั่ง `dropna()` เพื่อตัดแถวที่ข้อมูลไม่ครบถ้วนออก เพื่อความแม่นยำของโมเดล
    2. **การลบข้อมูลซ้ำ:** ใช้คำสั่ง `drop_duplicates()` เพื่อป้องกันไม่ให้โมเดลจดจำข้อมูลเดิมซ้ำซ้อน
    """)
    st.code("""
# ตัวอย่างโค้ดที่ใช้เตรียมข้อมูล
df = pd.read_csv('datasets/heart_data.csv')
df = df.dropna().drop_duplicates()
    """, language='python')

st.write("---")
st.write("**แหล่งอ้างอิง:** กระบวนการพัฒนาอ้างอิงมาตรฐานจาก Scikit-learn Library")
