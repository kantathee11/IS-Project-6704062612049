import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# สร้างโฟลเดอร์เก็บข้อมูล
os.makedirs('datasets', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. สร้าง Dataset แบบไม่สมบูรณ์ (มีค่าว่าง/ซ้ำ) ตามเงื่อนไขอาจารย์
data1 = pd.DataFrame({
    'Age': [25, 30, 45, np.nan, 30, 50, 22, 40, 35, 35],
    'Stress': [2, 5, 8, 3, 5, np.nan, 1, 4, 7, 7],
    'Target': [0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
})
data1.to_csv('datasets/data_imperfect.csv', index=False)

# 2. เตรียมข้อมูล (Data Cleaning)
df = data1.dropna().drop_duplicates()
X, y = df[['Age', 'Stress']], df['Target']

# 3. สร้างโมเดล 1: Ensemble (RF + LR + SVC)
m1 = VotingClassifier(estimators=[('rf', RandomForestClassifier()), ('lr', LogisticRegression()), ('svc', SVC(probability=True))], voting='soft').fit(X, y)

# 4. สร้างโมเดล 2: Neural Network
m2 = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000).fit(X, y)

# บันทึกไฟล์
pickle.dump(m1, open('models/ensemble_model.pkl', 'wb'))
pickle.dump(m2, open('models/nn_model.pkl', 'wb'))
print("สร้างไฟล์สำเร็จ!")