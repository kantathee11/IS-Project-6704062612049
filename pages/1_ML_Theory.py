import streamlit as st

# ตั้งค่าหน้ากระดาษ
st.set_page_config(page_title="ML Theory - 6704062612049", layout="wide")

st.title("📚 รายละเอียดการพัฒนาโมเดล Machine Learning")
st.write("---")

# 1. ข้อมูลที่ใช้ (Dataset Specification) [cite: 3, 4, 5]
st.header("1. ข้อมูลที่ใช้ในการพัฒนา (Dataset)")
st.subheader("ที่มาของข้อมูล:")
st.write("Dataset ชุดนี้สร้างโดย Generative AI (Gemini) เพื่อใช้เป็นข้อมูลจำลองในการศึกษาพยากรณ์ความเสี่ยงด้านสุขภาพตามข้อกำหนดโปรเจค [cite: 4]")

st.subheader("รายละเอียด Feature[cite: 5]:")
st.markdown("""
* **Age:** อายุของบุคคล (ตัวเลข整數) [cite: 5]
* **Cholesterol:** ระดับคอเลสเตอรอลในเลือด (ตัวเลข) [cite: 5]
* **Stress_Level:** ระดับความเครียดประเมินจากแบบสอบถาม (ระดับ 1-10) [cite: 5]
* **Target:** ผลลัพธ์การทำนาย (0 = สุขภาพปกติ, 1 = กลุ่มเสี่ยง) [cite: 5]
""")

# 2. การเตรียมข้อมูล (Data Preparation) [cite: 6, 7]
st.header("2. การเตรียมข้อมูล (Data Preparation)")
st.warning("ความไม่สมบูรณ์ของข้อมูล: ตรวจพบค่าว่าง (Missing Values) และข้อมูลซ้ำซ้อน (Duplicates) [cite: 6]")
st.write("ขั้นตอนการจัดการข้อมูล[cite: 7]:")
st.code("""
# ขั้นตอนการ Clean ข้อมูลด้วย Python
import pandas as pd

df = pd.read_csv('datasets/heart_data.csv')
df = df.dropna()           # ลบแถวที่มีค่าว่างออก
df = df.drop_duplicates()  # ลบข้อมูลที่ซ้ำซ้อน
""", language='python')

# 3. ทฤษฎีอัลกอริทึม (Algorithm Theory) [cite: 10, 11]
st.header("3. อัลกอริทึม Machine Learning แบบ Ensemble")
st.write("เลือกใช้เทคนิค Ensemble แบบ Voting Classifier ซึ่งประกอบด้วย 3 โมเดลหลัก[cite: 10]:")
st.markdown("""
1. **Random Forest:** อัลกอริทึมแบบกลุ่มต้นไม้ตัดสินใจ (Decision Trees) [cite: 11]
2. **Logistic Regression:** การวิเคราะห์การถดถอยเพื่อหาความน่าจะเป็น [cite: 11]
3. **SVC (Support Vector Machine):** การหาเส้นแบ่งกลุ่มข้อมูลที่มีประสิทธิภาพสูง [cite: 11]
""")

st.write("**แหล่งอ้างอิงข้อมูล:** Scikit-learn Open Source Library และ ข้อมูลจำลองจาก Gemini [cite: 17]")
