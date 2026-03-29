import streamlit as st

# 1. การตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ML Theory - 6704062612049", layout="wide", page_icon="🧪")

# 2. ส่วนหัว (Header)
st.title("🧪 Machine Learning Theory")
st.subheader("ทฤษฎีการพยากรณ์ความเสี่ยงสุขภาพด้วยเทคนิค Ensemble Learning")
st.write("การวิเคราะห์ความเสี่ยงสุขภาพในโปรเจคนี้ ใช้การรวมพลังของอัลกอริทึมที่หลากหลายเพื่อเพิ่มความแม่นยำในการทำนาย")
st.divider()

# 3. เนื้อหาหลักแบ่งเป็น 2 ฝั่ง (เหมือน Layout หน้า 2)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 🧬 Ensemble Learning (Voting)")
    st.write("""
    **Ensemble Learning** คือ เทคนิคการนำโมเดล Machine Learning หลายๆ ตัวมาทำงานร่วมกัน เพื่อหาข้อสรุปที่ดีที่สุด 
    ในโปรเจคนี้เราใช้ **Voting Classifier** ซึ่งทำหน้าที่เป็น 'กรรมการ' คอยรวบรวมผลโหวตจากโมเดล 3 ประเภท ดังนี้:
    """)
    
    # แสดงรายชื่อโมเดลที่ใช้ (เพิ่มส่วนนี้ตามที่คุณต้องการ)
    with st.expander("🔍 รายชื่อโมเดลที่ใช้ในระบบ (3 Algorithms)", expanded=True):
        st.markdown("""
        1. **Random Forest (Bagging):** เน้นลดความแปรปรวนและจัดการข้อมูลที่ซับซ้อน
        2. **Logistic Regression (Linear):** เน้นความเสถียรและความเป็นเหตุเป็นผลของข้อมูล
        3. **Support Vector Machine (Kernel):** เน้นการแยกแยะขอบเขตข้อมูลที่ชัดเจน
        """)

with col2:
    st.markdown("### 📊 ตัวแปรที่ใช้ในการวิเคราะห์ (Features)")
    st.write("ข้อมูลหลักที่นำมาใช้สอนโมเดลประกอบด้วย:")
    st.info("""
    - **Age (อายุ):** ปัจจัยพื้นฐานที่มีผลต่อความเสี่ยงสุขภาพ
    - **Cholesterol (คอเลสเตอรอล):** ค่าชี้วัดทางชีวภาพที่สำคัญ
    - **Stress Level (ระดับความเครียด):** ปัจจัยทางด้านสภาวะจิตใจและการใช้ชีวิต
    """)

st.write("---")

# 4. ส่วนอธิบายรายละเอียดของแต่ละโมเดล (เหมือนการเจาะลึกทฤษฎี)
st.markdown("### 🛠️ เจาะลึกการทำงานของแต่ละอัลกอริทึม")
tab1, tab2, tab3 = st.tabs(["Random Forest", "Logistic Regression", "Support Vector Machine"])

with tab1:
    st.write("**Random Forest** ใช้หลักการสร้างต้นไม้ตัดสินใจ (Decision Trees) จำนวนมาก แล้วนำคำตอบที่ได้มาหาค่าเฉลี่ย ช่วยป้องกันปัญหาการจำข้อมูลแม่นเกินไป (Overfitting)")
    

with tab2:
    st.write("**Logistic Regression** เป็นโมเดลทางสถิติที่พยากรณ์โอกาสที่จะเกิดเหตุการณ์ใดเหตุการณ์หนึ่ง (เช่น เสี่ยง หรือ ไม่เสี่ยง) โดยใช้ฟังก์ชัน Sigmoid เพื่อจำแนกประเภทข้อมูล")

with tab3:
    st.write("**Support Vector Machine (SVC)** ทำงานโดยการสร้างเส้นแบ่ง (Hyperplane) ที่พยายามแยกกลุ่มข้อมูลให้ห่างจากกันมากที่สุด เพื่อให้การจำแนกประเภทมีความแม่นยำสูงสุด")

st.write("---")
st.success("✅ โครงสร้างนี้ถูกจัดวางให้สอดคล้องกับหน้า Neural Network เพื่อความเป็นเอกภาพของโปรเจค")
