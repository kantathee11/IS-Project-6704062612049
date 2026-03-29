import streamlit as st

st.set_page_config(page_title="ML Theory - 6704062612049", layout="wide", page_icon="🧪")

st.title("🧪 ทฤษฎี Machine Learning แบบ Ensemble Learning")
st.write("ในโปรเจคนี้เราใช้เทคนิค **Voting Classifier** ซึ่งเป็นการรวมพลังของโมเดล 3 ประเภท เพื่อพยากรณ์ความเสี่ยงด้านสุขภาพ")
st.divider()

# อธิบายโมเดลทั้ง 3 ประเภท
col1, col2, col3 = st.columns(3)

with col1:
    st.header("1. Random Forest")
    st.markdown("""
    **ประเภท:** Bagging Ensemble
    - สร้าง 'ต้นไม้ตัดสินใจ' (Decision Trees) จำนวนมากมาทำงานขนานกัน
    - **จุดเด่น:** ช่วยลดความแปรปรวน (Variance) ของข้อมูล และป้องกันการจำข้อมูลแม่นเกินไป (Overfitting)
    - เหมาะสำหรับข้อมูลที่มีความสัมพันธ์ซับซ้อนและไม่ใช่เส้นตรง
    """)

with col2:
    st.header("2. Logistic Regression")
    st.markdown("""
    **ประเภท:** Linear Model
    - คำนวณความน่าจะเป็นโดยใช้ฟังก์ชันทางคณิตศาสตร์ (Sigmoid)
    - **จุดเด่น:** ให้ผลลัพธ์ที่เสถียรและตีความหมายได้ง่าย เป็นฐานการพยากรณ์ที่ดีหากข้อมูลมีแนวโน้มเป็นเหตุเป็นผลชัดเจน
    - ช่วยให้โมเดลโดยรวมมีความสมดุล (Bias-Variance Tradeoff)
    """)

with col3:
    st.header("3. Support Vector Machine (SVC)")
    st.markdown("""
    **ประเภท:** Kernel-based Model
    - พยายามหา 'เส้นแบ่ง' หรือ 'Hyperplane' ที่กว้างที่สุดเพื่อแยกกลุ่มข้อมูล
    - **จุดเด่น:** แม่นยำสูงในการแยกแยะข้อมูลที่มีขอบเขตคาบเกี่ยวกัน (Decision Boundary)
    - ช่วยเพิ่มประสิทธิภาพในการจำแนกกลุ่มเสี่ยงและกลุ่มปกติให้ชัดเจนขึ้น
    """)

st.write("---")

# อธิบายหลักการ Voting
st.subheader("🤝 ทำไมต้องใช้ Ensemble (Voting)?")
st.info("""
การทำ **Ensemble** แบบ Voting เปรียบเสมือนการปรึกษา 'คณะกรรมการ' โดยให้นำคำทำนายจากทั้ง 3 โมเดลข้างต้นมา 'โหวต' กัน (ในโปรเจคนี้ใช้ Soft Voting คือการเฉลี่ยความน่าเชื่อถือ) 
วิธีนี้จะช่วยแก้จุดอ่อนของโมเดลเดี่ยวๆ ทำให้ระบบพยากรณ์มีความแม่นยำและน่าเชื่อถือมากขึ้นครับ
""")

st.write("---")
st.caption("จัดทำโดย: รหัสนักศึกษา 6704062612049")
