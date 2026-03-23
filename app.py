import streamlit as st

st.set_page_config(page_title="IS Project 6704062612049", layout="wide", page_icon="📊")

# ส่วนหัวของหน้าเว็บ
st.title("🌟 ระบบวิเคราะห์ข้อมูลด้วย AI และ Machine Learning")
st.subheader("โปรเจครายวิชา IS | รหัสนักศึกษา: 6704062612049")
st.write("---")

# ใช้ Columns เพื่อจัดระเบียบหน้าแรก
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 👨‍💻 เกี่ยวกับโปรเจคนี้
    โปรเจคนี้จัดทำขึ้นเพื่อสาธิตการนำข้อมูลที่มีความไม่สมบูรณ์ (Imperfect Data) มาผ่านกระบวนการเตรียมข้อมูล 
    และนำไปพัฒนาต่อยอดเป็นโมเดลพยากรณ์ 2 รูปแบบ คือ:
    1. **Machine Learning (Ensemble):** การรวมพลังของหลายอัลกอริทึม
    2. **Neural Network:** โครงข่ายประสาทเทียมที่ออกแบบเอง
    
    👈 **กรุณาเลือกเมนูจากแถบด้านซ้ายเพื่อเริ่มต้นใช้งาน**
    """)

with col2:
    st.info("📌 **สถานะระบบ:** ออนไลน์ (Online)")
    st.success("✅ **Dataset:** พร้อมใช้งาน")
    st.warning("⚠️ **Model:** โหลดสำเร็จ")

st.image("https://images.unsplash.com/photo-1551288049-bbbda536ad31?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", use_container_width=True)
