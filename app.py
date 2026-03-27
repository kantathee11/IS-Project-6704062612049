import streamlit as st

# 1. การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="AI Dashboard - 6704062612049", 
    layout="wide", 
    page_icon="📊"
)

# 2. ส่วนหัวของเว็บไซต์
st.title("🌟 AI & Machine Learning Dashboard")
st.subheader("ระบบวิเคราะห์และพยากรณ์ข้อมูล | รหัสนักศึกษา: 6704062612049")
st.write("โครงการนี้พัฒนาขึ้นเพื่อนำเสนอการประยุกต์ใช้โมเดลการเรียนรู้ของเครื่องในด้านสุขภาพและการวิเคราะห์ข้อความ")
st.divider()

# 3. เมนูทางลัด (Dashboard Cards)
col_a, col_b = st.columns(2)

with col_a:
    with st.container(border=True):
        st.markdown("### 🧪 Machine Learning Section")
        st.write("พยากรณ์ความเสี่ยงสุขภาพพนักงาน โดยใช้เทคนิค **Ensemble Learning (Voting)** ที่รวมจุดเด่นของ 3 โมเดลเข้าด้วยกัน")
        st.info("📍 ดูทฤษฎีได้ที่หน้า 1 | ทดลองใช้ได้ที่หน้า 3")

with col_b:
    with st.container(border=True):
        st.markdown("### 🧠 Neural Network Section")
        st.write("วิเคราะห์ความรู้สึกจากรีวิว (Sentiment Analysis) โดยใช้ **Multi-layer Perceptron (MLP)** ซึ่งเป็นโครงข่ายประสาทเทียมระดับพื้นฐาน")
        st.info("📍 ดูทฤษฎีได้ที่หน้า 2 | ทดลองใช้ได้ที่หน้า 4")

st.write("---")

# 4. ส่วน Workflow (สร้างด้วย Code เพื่อป้องกันภาพไม่โหลด)
st.markdown("#### 🛠️ กระบวนการพัฒนาโมเดล (ML & NN Workflow)")

# สร้าง Step การทำงานแบบตารางเพื่อให้ดูเป็นลำดับขั้นตอน
step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown("📂 **1. Data Collection**")
    st.caption("จัดเตรียมชุดข้อมูล CSV จากการสำรวจและรวบรวมรีวิว")

with step2:
    st.markdown("🧹 **2. Preprocessing**")
    st.caption("ทำความสะอาดข้อมูล, ลบค่าว่าง และแปลงข้อความเป็นตัวเลข (TF-IDF)")

with step3:
    st.markdown("⚙️ **3. Model Training**")
    st.caption("ฝึกสอนโมเดลด้วย Voting Classifier และ Neural Network")

with step4:
    st.markdown("🚀 **4. Deployment**")
    st.caption("นำเสนอผลลัพธ์ผ่าน Streamlit และประมวลผลแบบ Real-time")

# 5. ส่วนทฤษฎีเพิ่มเติม (Expander)
st.write("")
with st.expander("📝 ข้อมูลเกี่ยวกับเทคนิคที่ใช้ในโปรเจคนี้"):
    st.markdown("""
    - **Ensemble (Voting):** ใช้การรวมผลจาก Random Forest, Logistic Regression และ SVC เพื่อลด Error
    - **Neural Network (MLP):** ใช้ Hidden Layers 2 ชั้น (16, 8) เพื่อเรียนรู้ความสัมพันธ์ที่ซับซ้อนของคำศัพท์
    - **TF-IDF Vectorizer:** ใช้แปลงข้อความภาษาอังกฤษให้เป็นค่าน้ำหนักเชิงสถิติเพื่อให้ AI เข้าใจความหมาย
    """)

st.write("---")
st.success("👈 กรุณาเลือกหัวข้อที่ต้องการตรวจสอบจาก Sidebar (เมนูด้านซ้าย)")

import streamlit as st

def show_documentation():
    with open("DOCUMENTATION.md", "r", encoding="utf-8") as f:
        doc_content = f.read()
    st.markdown(doc_content)

# เรียกใช้ฟังก์ชันในหน้าเว็บที่คุณต้องการ
show_documentation()
