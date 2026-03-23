import streamlit as st

st.set_page_config(page_title="ทฤษฎี ML - 6704062612049", layout="wide")

st.title("📚 รายละเอียดการพัฒนาโมเดล Machine Learning")
st.write("---")

# a. ระบุที่มาของ Dataset
st.header("A. ที่มาของข้อมูล (Dataset Source)")
st.write("ข้อมูลชุดนี้เป็นข้อมูลจำลอง (Synthetic Data) ที่สร้างขึ้นโดย Generative AI เพื่อจำลองสถานการณ์การวิเคราะห์ความเสี่ยงด้านสุขภาพของพนักงานในองค์กร")

# b. อธิบาย feature ของ Dataset
st.header("B. คำอธิบายตัวแปร (Features)")
st.markdown("""
ข้อมูลชุด 'heart_data.csv' ประกอบด้วยตัวแปรหลักดังนี้:
* **Age:** อายุของพนักงาน (ข้อมูลตัวเลข)
* **Cholesterol:** ระดับคอเลสเตอรอลในเลือด
* **Stress_Level:** ระดับความเครียดประเมินจากแบบสอบถาม (1-10)
* **Target:** ผลลัพธ์การทำนาย (0 = ปกติ, 1 = เสี่ยงโรค)
""")

# c. ความไม่สมบูรณ์และการเตรียมข้อมูล
st.header("C. การจัดการข้อมูลที่ไม่สมบูรณ์ (Data Preparation)")
st.warning("Dataset นี้ถูกออกแบบให้มีค่าว่าง (Missing Values) และข้อมูลซ้ำ (Duplicates)")
st.write("ขั้นตอนการเตรียมข้อมูล:")
st.markdown("""
1. **ลบค่าว่าง:** ใช้คำสั่ง `dropna()` เพื่อตัดแถวที่มีข้อมูลไม่ครบถ้วนออก
2. **ลบข้อมูลซ้ำ:** ใช้คำสั่ง `drop_duplicates()` เพื่อป้องกันโมเดลเรียนรู้จากข้อมูลซ้ำซ้อน
""")

st.write("---")
st.header("D. ทฤษฎีอัลกอริทึมและแหล่งอ้างอิง")
st.write("โมเดลแบบ **Ensemble (Voting)**: เป็นการรวมการตัดสินใจจาก Random Forest, Logistic Regression และ SVC")
st.write("**แหล่งอ้างอิง:** Scikit-learn Documentation")
