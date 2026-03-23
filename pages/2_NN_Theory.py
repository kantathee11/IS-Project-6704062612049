import streamlit as st
import pandas as pd

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="NN Theory - 6704062612049", layout="wide", page_icon="🧠")

st.title("🧠 รายละเอียดการพัฒนาโมเดล Neural Network")
st.write("---")

# ส่วนแสดงสถิติแบบ Metric
m1, m2, m3 = st.columns(3)
m1.metric("Dataset Name", "review_data.csv")
m2.metric("Model Type", "Neural Network (MLP)")
m3.metric("Architecture", "16-8-1 Neurons")

# a. ระบุที่มาของ Dataset
with st.expander("📌 A. ที่มาของข้อมูล (Dataset Source)", expanded=True):
    st.write("ข้อมูลชุดนี้เป็น **ข้อมูลจำลอง (Synthetic Data)** ที่สร้างขึ้นโดย Generative AI เพื่อใช้ในการวิเคราะห์ความรู้สึก (Sentiment Analysis) จากข้อความรีวิวภาษาอังกฤษ")

# b. อธิบาย feature ของ Dataset
with st.expander("📊 B. คำอธิบายตัวแปร (Features)", expanded=True):
    st.write("ข้อมูลชุด 'review_data.csv' ประกอบด้วยตัวแปรหลัก ดังนี้:")
    st.markdown("""
    * **Review_Text:** ข้อความรีวิวจากลูกค้า (ข้อมูลแบบข้อความไม่ระบุรูปแบบ)
    * **Sentiment:** ความรู้สึกที่ได้จากรีวิว (0 = เชิงลบ/Negative, 1 = เชิงบวก/Positive)
    """)

# c. ความไม่สมบูรณ์และการเตรียมข้อมูล
with st.expander("🛠️ C. การจัดการข้อมูลที่ไม่สมบูรณ์ (Data Preparation)", expanded=True):
    st.error("⚠️ ตรวจพบความไม่สมบูรณ์: ข้อมูลมีอักขระพิเศษ HTML Tags และเว้นวรรคส่วนเกิน")
    st.write("**ขั้นตอนการเตรียมข้อมูล:**")
    st.markdown("""
    1. **Text Cleaning:** ใช้ Regex ในการลบ HTML Tags เช่น `<br>`, `<p>` ออกจากข้อความรีวิว
    2. **Normalization:** ปรับข้อความเป็นตัวพิมพ์เล็กทั้งหมด (Lowercasing) เพื่อให้โมเดลจำคำศัพท์ได้แม่นยำขึ้น
    """)
    st.code("""
# ตัวอย่างโค้ดที่ใช้เตรียมข้อมูลข้อความ
df['Review_Text'] = df['Review_Text'].str.replace(r'<[^>]*>', '', regex=True).str.lower()
    """, language='python')

st.write("---")
st.write("**แหล่งอ้างอิง:** ทฤษฎีโครงข่ายประสาทเทียม (Artificial Neural Networks) เบื้องต้น")
