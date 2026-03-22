import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.neural_network import MLPClassifier

st.title("🧠 ทดสอบโมเดล Neural Network")

if not os.path.exists('models/nn_model.pkl'):
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('datasets/heart_data.csv')
    df = df.dropna().drop_duplicates()
    X = df[['Age', 'Cholesterol', 'Stress_Level']]
    y = df['Target']
    
    # ออกแบบโครงสร้าง Neural Network เองตามเงื่อนไข
    m2 = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000)
    m2.fit(X, y)
    pickle.dump(m2, open('models/nn_model.pkl', 'wb'))

model = pickle.load(open('models/nn_model.pkl', 'rb'))

age = st.number_input("ระบุอายุ", value=25)
chol = st.number_input("ระบุคอเลสเตอรอล", value=180)
stress = st.number_input("ระบุความเครียด (1-10)", value=3)

if st.button("วิเคราะห์ด้วย Neural Network"):
    pred = model.predict([[age, chol, stress]])
    st.write(f"ผลการวิเคราะห์: {'กลุ่มเสี่ยง' if pred[0] == 1 else 'กลุ่มปกติ'}")
