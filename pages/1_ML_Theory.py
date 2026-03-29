import streamlit as st

# 1. การตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ML Theory - 6704062612049", layout="wide", page_icon="🧠")

# 2. ส่วนหัวหลัก (Header)
st.title("🧠 รายละเอียดการพัฒนาโมเดล Machine Learning")
st.write("---")

# 3. ส่วนแสดงข้อมูลเบื้องต้น (Summary Cards)
col_set, col_type, col_arch = st.columns(3)
with col_set:
    st.caption("Dataset Name")
    st.subheader("heart_data.csv")

with col_type:
    st.caption("Model Type")
    st.subheader("Ensemble Learning")

with col_arch:
    st.caption("Architecture")
    st.subheader("Voting Classifier (3 Models)")

# 4. รายละเอียดเนื้อหาโดยใช้ Expander (รูปแบบเดียวกับหน้า NN)
# ส่วน A: ที่มาของข้อมูล
with st.expander("📌 A. ที่มาของข้อมูล (Dataset Source)", expanded=True):
    st.write("ข้อมูลชุดนี้เป็น **ข้อมูลจำลอง (Synthetic Data)** ที่สร้างขึ้นเพื่อให้มีความใกล้เคียงกับสถานการณ์จริง เพื่อใช้ในการพยากรณ์ความเสี่ยงสุขภาพ [cite: 6]")

# ส่วน B: คำอธิบายตัวแปร
with st.expander("📊 B. คำอธิบายตัวแปร (Features)"):
    st.write("ข้อมูลชุด 'heart_data.csv' ประกอบด้วยตัวแปรหลัก ดังนี้: [cite: 6]")
    st.markdown("""
    * **Age:** ช่วงอายุของผู้เข้ารับการตรวจ [cite: 8]
    * **Cholesterol:** ระดับคอเลสเตอรอลในเลือด [cite: 8]
    * **Stress Level:** ระดับความเครียดจากการใช้ชีวิต [cite: 8]
    * **Target:** ผลการพยากรณ์ (0 = ปกติ, 1 = มีความเสี่ยง) 
    """)

# ส่วน C: รายชื่อโมเดลที่ใช้ (เพิ่มตามที่คุณต้องการ)
with st.expander("🛠️ C. รายชื่อโมเดลที่ใช้ในระบบ (Ensemble Models)", expanded=True):
    st.write("โปรเจกต์นี้เลือกใช้เทคนิค **Voting Classifier** เพื่อรวมพลังของ 3 โมเดลที่แตกต่างกัน เพื่อลดความลำเอียงและเพิ่มความเสถียร: ")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.info("**1. Random Forest**")
        st.caption("ใช้หลักการ Bagging สร้างต้นไม้ตัดสินใจหลายต้นเพื่อลดความแปรปรวน [cite: 13]")
    
    with col_m2:
        st.info("**2. Logistic Regression**")
        st.caption("ใช้การหาความสัมพันธ์เชิงเส้นเพื่อระบุโอกาสเกิดความเสี่ยงแบบเป็นเหตุเป็นผล [cite: 14]")
        
    with col_m3:
        st.info("**3. Support Vector Machine (SVC)**")
        st.caption("ใช้การสร้างขอบเขต (Margin) ที่กว้างที่สุดเพื่อแยกกลุ่มข้อมูลสุขภาพ [cite: 15]")

# ส่วน D: การจัดการข้อมูล (Data Preparation)
with st.expander("⚙️ D. การจัดการข้อมูลและการเตรียมความพร้อม"):
    st.error("⚠️ ตรวจพบความไม่สมบูรณ์: ข้อมูลมีค่าว่างและข้อมูลซ้ำซ้อน ")
    st.write("**ขั้นตอนการเตรียมข้อมูล:**")
    st.markdown("""
    1. **Data Cleansing:** ทำการลบข้อมูลซ้ำซ้อน (Drop Duplicates) และจัดการกับค่าว่าง (Dropna) 
    2. **Scaling:** ปรับสเกลตัวเลขในข้อมูลสุขภาพให้พร้อมสำหรับการประมวลผล 
    """)

st.write("---")
st.caption("รหัสนักศึกษา: 6704062612049 | ระบบพยากรณ์ความเสี่ยงสุขภาพด้วย AI [cite: 2, 3]")
