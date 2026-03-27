import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Ensemble Learning - 6704062612049", layout="wide", page_icon="🧪")

st.title("🧪 ระบบพยากรณ์ความเสี่ยงสุขภาพด้วยเทคนิค Ensemble")
st.write("การวิเคราะห์นี้ใช้การรวมพลังของ 3 โมเดล (Random Forest, Logistic Regression, SVC) เพื่อผลลัพธ์ที่แม่นยำที่สุด")
st.write("---")

# 2. ส่วนข้อมูลสถิติ
try:
    df_health = pd.read_csv('datasets/heart_data.csv').dropna().drop_duplicates()
    st.subheader("📊 ตารางสรุปสถิติชุดข้อมูล")
    # แสดงสถิติพื้นฐาน
    st.table(df_health[['Age', 'Cholesterol', 'Stress_Level']].describe().loc[['mean', 'min', 'max']])
except Exception as e:
    st.error("ไม่พบไฟล์ datasets/heart_data.csv")
    st.stop()

st.write("---")

# 3. ส่วนการ Train Model (ใช้ 3 โมเดลตามเงื่อนไข)
model_path = 'models/ensemble_model.pkl'
if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists(model_path):
    with st.status("กำลังฝึกสอนโมเดล Ensemble (3 Algorithms)..."):
        X = df_health[['Age', 'Cholesterol', 'Stress_Level']]
        y = df_health['Target']
        
        # 3 โมเดลที่แตกต่างกัน
        m1 = RandomForestClassifier(n_estimators=100, random_state=42)
        m2 = LogisticRegression(random_state=42)
        m3 = SVC(probability=True, random_state=42)
        
        # รวมเข้าด้วยกันเป็น Voting Classifier
        ensemble_model = VotingClassifier(
            estimators=[('rf', m1), ('lr', m2), ('svc', m3)],
            voting='soft' # ใช้ความน่าเชื่อถือของแต่ละโมเดลมาเฉลี่ยกัน
        )
        
        ensemble_model.fit(X, y)
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble_model, f)
        st.success("สร้างระบบ Ensemble เสร็จสมบูรณ์!")

# โหลดโมเดลมาใช้
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 4. ส่วนรับข้อมูลและพยากรณ์
with st.container(border=True):
    st.subheader("📝 กรอกข้อมูลพนักงาน")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("ระบุอายุ (ปี):", 1, 100, 30)
    with col2:
        chol = st.number_input("ระบุคอเลสเตอรอล:", 100, 500, 200)
    stress = st.select_slider("ระดับความเครียด (1-10):", options=list(range(1, 11)), value=5)

    if st.button("🚀 วิเคราะห์ความเสี่ยง", use_container_width=True):
        pred = model.predict([[age, chol, stress]])
        st.write("---")
        if pred[0] == 1:
            st.error("### ⚠️ ผลการวิเคราะห์: มีความเสี่ยงต่อสุขภาพ")
            st.write("คำแนะนำ: ควรปรึกษาแพทย์เพื่อตรวจเช็คอย่างละเอียด")
        else:
            st.success("### ✅ ผลการวิเคราะห์: สุขภาพปกติดี")
            st.balloons()

st.write("---")
st.info("💡 ข้อมูลทางเทคนิค: ระบบใช้การโหวตแบบ Soft Voting จาก 3 อัลกอริทึมที่แตกต่างกันเพื่อลดความลำเอียงของโมเดลใดโมเดลหนึ่ง")
