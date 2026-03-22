import streamlit as st

st.set_page_config(page_title="NN Theory - 6704062612049", layout="wide")

st.title("🧠 รายละเอียดการพัฒนาโมเดล Neural Network")
st.write("---")

# ส่วนที่ 1: ข้อมูล
st.header("1. ข้อมูลที่ใช้พัฒนา")
st.write("ใช้ Dataset 'Heart Data' เช่นเดียวกับโมเดลแรก โดยนำค่า Age, Cholesterol และ Stress_Level มาเป็น Input ของโครงข่ายประสาทเทียม")

# ส่วนที่ 2: โครงสร้างโมเดล (เงื่อนไขข้อ 3b)
st.header("2. การออกแบบโครงสร้างโมเดล (Model Architecture)")
st.write("โมเดลนี้เป็นประเภท **Multi-Layer Perceptron (MLP)** ที่ออกแบบจำนวน Layer และ Neuron ให้เหมาะสมกับขนาดข้อมูล:")
st.image("https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg", width=400, caption="แผนภาพโครงสร้าง Neural Network")

st.markdown("""
- **Input Layer:** รับข้อมูล 3 ตัวแปร (Age, Cholesterol, Stress)
- **Hidden Layer 1:** 16 Neurons (ใช้ Activation Function: ReLU)
- **Hidden Layer 2:** 8 Neurons (ใช้ Activation Function: ReLU)
- **Output Layer:** 1 Neuron (ทำนายผลลัพธ์เป็น 0 หรือ 1)
""")

# ส่วนที่ 3: ขั้นตอนการพัฒนา
st.header("3. ขั้นตอนการพัฒนาโมเดล")
st.markdown("""
1. **Normalization:** ปรับช่วงของข้อมูลให้อยู่ในช่วงเดียวกันเพื่อให้ AI เรียนรู้ได้ไวขึ้น
2. **Training:** ใช้ตัวแก้ปัญหา (Solver) แบบ 'Adam' ซึ่งเป็นที่นิยมในงาน Deep Learning
3. **Iteration:** กำหนดรอบการเรียนรู้ (Max Iteration) ไว้ที่ 1,000 รอบ
""")

st.header("4. แหล่งอ้างอิงข้อมูล")
st.write("- ข้อมูลจำลองสร้างโดย Generative AI")
st.write("- ทฤษฎี Neural Network จากหนังสือ 'Python Machine Learning'")
