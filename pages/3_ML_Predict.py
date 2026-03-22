import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

st.title("🧪 ทดสอบโมเดล Machine Learning (Ensemble)")

# ส่วนนี้จะเช็คว่าถ้าไม่มีโมเดล ให้สร้างใหม่จาก Dataset ที่คุณเพิ่งอัปโหลด
if not os.path.exists('models/ensemble_model.pkl'):
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('datasets/heart_data.csv')
    df = df.dropna().drop_duplicates() # ทำ Data Cleaning ตามเงื่อนไข
    X = df[['Age', 'Cholesterol', 'Stress_Level']]
    y = df['Target']
    
    # สร้าง Ensemble จาก 3 ประเภท: Random Forest, Logistic, SVM
    m1 = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier()), 
        ('lr', LogisticRegression()), 
        ('svc', SVC(probability=True))
    ], voting='soft')
    m1.fit(X, y)
    pickle.dump(m1, open('models/ensemble_model.pkl', 'wb'))

# โหลดโมเดลมาใช้งาน
model = pickle.load(open('models/ensemble_model.pkl', 'rb'))

# สร้างช่องรับข้อมูล
age = st.number_input("อายุ (Age)", value=30)
chol = st.number_input("คอเลสเตอรอล (Cholesterol)", value=200)
stress = st.slider("ระดับความเครียด (Stress Level)", 1, 10, 5)

if st.button("ประมวลผลการทำนาย"):
    result = model.predict([[age, chol, stress]])
    if result[0] == 1:
        st.error("ผลทำนาย: มีความเสี่ยงต่อสุขภาพ")
    else:
        st.success("ผลทำนาย: สุขภาพปกติ")
