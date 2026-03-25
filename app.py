import streamlit as st

st.set_page_config(page_title="IS Project - 6704062612049", layout="wide", page_icon="📊")

# ส่วนหัว
st.title("🌟 AI & Machine Learning Dashboard")
st.subheader("การวิเคราะห์ข้อมูลด้วยโมเดลพยากรณ์ | รหัสนักศึกษา: 6704062612049")
st.divider()

# การจัดเลย์เอาต์หน้าแรก
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("### 🧪 ส่วนที่ 1: Machine Learning")
        st.write("การพยากรณ์ความเสี่ยงด้านสุขภาพโดยใช้เทคนิค **Ensemble Learning (Voting)**")
        st.info("หน้า 1: ทฤษฎี ML | หน้า 3: ทดสอบพยากรณ์")

with col2:
    with st.container(border=True):
        st.markdown("### 🧠 ส่วนที่ 2: Neural Network")
        st.write("การวิเคราะห์ความรู้สึกจากข้อความ (Sentiment Analysis) โดยใช้ **MLP Classifier**")
        st.info("หน้า 2: ทฤษฎี NN | หน้า 4: ทดสอบพยากรณ์")

st.write("---")
st.markdown("#### 🛠️ กระบวนการพัฒนา (Workflow)")
st.image("https://miro.medium.com/v2/resize:fit:1400/1*Gu0pU0S55-m8v_H5m7qf7A.png", caption="Machine Learning Workflow", use_container_width=True)

st.success("👈 กรุณาเลือกหัวข้อที่ต้องการตรวจสอบจาก Sidebar ด้านซ้าย")
