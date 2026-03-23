import streamlit as st

st.set_page_config(page_title="ทฤษฎี NN - 6704062612049", layout="wide")

st.title("🧠 รายละเอียดการพัฒนาโมเดล Neural Network")
st.write("---")

st.header("A. ที่มาของข้อมูล (Dataset Source)")
st.write("ข้อมูลชุดนี้เป็นข้อมูลจำลอง (Synthetic Data) ที่สร้างขึ้นโดย Generative AI เพื่อใช้ในการวิเคราะห์ความรู้สึก (Sentiment Analysis) จากข้อความรีวิว")

st.header("B. คำอธิบายตัวแปร (Features)")
st.markdown("""
ข้อมูลชุด 'review_data.csv' ประกอบด้วยตัวแปรหลักดังนี้:
* **Review_Text:** ข้อความรีวิวจากลูกค้า (Unstructured Data)
* **Sentiment:** ความรู้สึกที่ได้จากรีวิว (0 = ลบ, 1 = บวก)
""")

st.header("C. การจัดการข้อมูลที่ไม่สมบูรณ์ (Data Preparation)")
st.warning("ข้อมูลรีวิวมีความสกปรกจากอักขระพิเศษ HTML Tags และเว้นวรรคส่วนเกิน")
st.write("ขั้นตอนการเตรียมข้อมูล:")
st.markdown("""
1. **Text Cleaning:** ลบ HTML Tags เช่น `<br>`, `<p>` ออกจากข้อความ
2. **Normalization:** ปรับข้อความเป็นตัวพิมพ์เล็กทั้งหมดเพื่อความแม่นยำในการประมวลผล
""")

st.write("---")
st.header("D. ทฤษฎีอัลกอริทึมและแหล่งอ้างอิง")
st.write("โมเดลแบบ **Neural Network (MLP)**: ออกแบบโครงสร้าง Hidden Layers (16, 8) เพื่อเรียนรู้แพทเทิร์นของคำ")
st.write("**แหล่งอ้างอิง:** หลักการประมวลผลภาษาธรรมชาติ (NLP) เบื้องต้น")
