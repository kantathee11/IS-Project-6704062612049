import streamlit as st
import pickle

st.title("🧪 ทดสอบ Neural Network")
model = pickle.load(open('models/nn_model.pkl', 'rb'))

age = st.number_input("ใส่อายุ", value=30)
stress = st.number_input("ใส่ระดับความเครียด", value=5)

if st.button("ประมวลผลด้วย AI"):
    res = model.predict([[age, stress]])
    st.success(f"AI วิเคราะห์ว่า: {'เสี่ยงภาวะ Burnout' if res[0]==1 else 'ปกติ'}")