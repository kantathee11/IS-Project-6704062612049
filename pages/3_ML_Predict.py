import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

st.set_page_config(page_title="วิเคราะห์ ML - 6704062612049")
st.title("🧪 ทดสอบระบบพยากรณ์สุขภาพ (Ensemble)")

# ล็อกค่าสุ่มด้วย random_state=42 เพื่อความแม่นยำและผลลัพธ์ที่คงที่
if not os.path.exists('models/ensemble_model.pkl'):
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('datasets/heart_data.csv').dropna().drop_duplicates()
    X, y = df[['Age', 'Cholesterol', 'Stress_Level']], df['Target']
    
    # ปรับจูนโมเดลให้ฉลาดขึ้น
    m1 = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)), 
        ('lr', LogisticRegression(random_state=42)), 
        ('svc', SVC(probability=True, random_state=42))
    ], voting='soft').fit(X, y)
    
    pickle.dump(m1, open('models/ensemble_model.pkl', 'wb'))

model = pickle.load(open('models/ensemble_model.pkl', 'rb'))

st.write("---")
age = st.number_input("อายุ", value=30)
chol = st.number_input("คอเลสเตอรอล", value=200)
stress = st.slider("ระดับความเครียด", 1, 10, 5)

if st.button("วิเคราะห์ผล"):
    pred = model.predict([[age, chol, stress]])
    if pred[0] == 1:
        st.error("⚠️ ผล: มีความเสี่ยงต่อสุขภาพ")
    else:
        st.success("✅ ผล: สุขภาพปกติ")
