import streamlit as st
import pickle

st.title("🧪 ทดสอบ Machine Learning")
model = pickle.load(open('models/ensemble_model.pkl', 'rb'))

age = st.slider("อายุ", 20, 60, 30)
stress = st.slider("ระดับความเครียด (1-10)", 1, 10, 5)

if st.button("ทำนายผล"):
    res = model.predict([[age, stress]])
    st.write(f"ผลลัพธ์: {'มีภาวะ Burnout' if res[0]==1 else 'ยังไหวอยู่!'}")