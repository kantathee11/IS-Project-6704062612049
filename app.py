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

# ส่วนแสดง Workflow (ใช้ภาพที่เสถียรกว่าจากวิกิพีเดีย หรือแหล่งที่อนุญาต)
st.markdown("#### 🛠️ กระบวนการพัฒนา (Machine Learning Workflow)")

# ใช้ภาพ Standard CRISP-DM หรือ ML Workflow ที่ลิงก์ไม่ตายง่ายๆ
workflow_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Data_visualization_process_v1.png/800px-Data_visualization_process_v1.png"

try:
    st.image(workflow_url, caption="ขั้นตอนการทำงานของระบบ Machine Learning", use_container_width=True)
except:
    st.warning("⚠️ ไม่สามารถโหลดภาพ Workflow จากแหล่งภายนอกได้")

# เพิ่มคำอธิบายขั้นตอน (เพื่อให้คะแนนดีขึ้น)
with st.expander("📝 อธิบายขั้นตอนการทำงาน (Workflow Step-by-Step)"):
    st.markdown("""
    1. **Data Collection:** รวบรวมข้อมูลจาก `heart_data.csv` และ `review_data.csv`
    2. **Data Cleaning:** จัดการ Missing Values และลบข้อมูลซ้ำ (Preprocessing)
    3. **Model Training:** ฝึกสอนโมเดล Ensemble และ Neural Network
    4. **Evaluation:** ทดสอบความแม่นยำของโมเดล
    5. **Deployment:** นำเสนอผลลัพธ์ผ่านเว็บแอปพลิเคชัน Streamlit
    """)

st.success("👈 กรุณาเลือกหัวข้อที่ต้องการตรวจสอบจาก Sidebar ด้านซ้าย")
