import streamlit as st
st.title("📚 ทฤษฎี Machine Learning")
st.header("1. ข้อมูลที่ใช้")
st.write("ใช้ Dataset 'Employee Health' ที่สร้างขึ้นเองเพื่อการศึกษา [cite: 4]")
st.header("2. การเตรียมข้อมูล [cite: 7]")
st.write("- ลบค่าว่าง (Missing Values) ")
st.write("- ลบข้อมูลซ้ำซ้อน (Duplicates)")
st.header("3. อัลกอริทึม")
st.write("ใช้ Ensemble Learning แบบ Voting ที่รวม Random Forest, Logistic Regression และ SVC เข้าด้วยกัน [cite: 10, 11]")
