import streamlit as st

st.set_page_config(page_title="ML Theory - 6704062612049", layout="wide")

st.title("📚 รายละเอียดการพัฒนาโมเดล Machine Learning")
st.write("---")

# ส่วนที่ 1: ข้อมูล (Dataset)
st.header("1. ข้อมูลที่ใช้ในการพัฒนา (Dataset)")
st.subheader("ที่มาของข้อมูล:")
st.write("Dataset ชุดนี้สร้างโดย Generative AI (Gemini) เพื่อใช้เป็นข้อมูลจำลองในการศึกษาพยากรณ์ความเสี่ยงด้านสุขภาพ")

st.subheader("รายละเอียด Feature:")
st.markdown("""
- **Age:** อายุของบุคคล (ตัวเลข)
- **Cholesterol:** ระดับคอเลสเตอรอลในเลือด (ตัวเลข)
- **Stress_Level:** ระดับความเครียดประเมินจากแบบสอบถาม (ระดับ 1-10)
- **Target:** ผลลัพธ์การทำนาย (0 = สุขภาพปกติ, 1 = กลุ่มเสี่ยง)
""")

# ส่วนที่ 2: การเตรียมข้อมูล (สำคัญมากตามเงื่อนไขข้อ 1c และ 2)
st.header("2. การเตรียมข้อมูล (Data Preparation)")
st.error("ความไม่สมบูรณ์ของข้อมูลที่พบ: ข้อมูลชุดนี้มีค่าว่าง (Missing Values) และข้อมูลซ้ำซ้อน (Duplicates)")
st.write("ขั้นตอนการ Clean ข้อมูล:")
st.code("""
# โค้ดที่ใช้ในการทำความสะอาดข้อมูล
df = pd.read_csv('datasets/heart_data.csv')
df = df.dropna()           # ลบแถวที่มีค่าว่างออก
df = df.drop_duplicates()  # ลบแถวที่มีข้อมูลซ้ำซ้อนกัน
""")

# ส่วนที่ 3: ทฤษฎีอัลกอริทึม (เงื่อนไขข้อ 3a)
st.header("3. อัลกอริทึม Machine Learning แบบ Ensemble")
st.write("โมเดลนี้ใช้เทคนิค **Voting Classifier** ซึ่งเป็นการรวมพลัง
