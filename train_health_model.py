import pandas as pd
import pickle
import os
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 1. โหลดข้อมูล
print("กำลังโหลดข้อมูล heart_data.csv...")
df = pd.read_csv('datasets/heart_data.csv')

# 2. ทำความสะอาดข้อมูล (Data Cleansing)
df = df.dropna().drop_duplicates()

# 3. เตรียม Feature และ Target
X = df[['Age', 'Cholesterol', 'Stress_Level']]
y = df['Target']

# 4. สร้างโมเดลแบบ Ensemble (3 Algorithms)
print("กำลังฝึกสอนโมเดล Ensemble (RF, LR, SVC)...")
m1 = RandomForestClassifier(n_estimators=100, random_state=42)
m2 = LogisticRegression(random_state=42)
m3 = SVC(probability=True, random_state=42)

ensemble_model = VotingClassifier(
    estimators=[('rf', m1), ('lr', m2), ('svc', m3)],
    voting='soft'
)

# 5. ฝึกสอนโมเดล
ensemble_model.fit(X, y)

# 6. บันทึกโมเดลลงโฟลเดอร์ models
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

print("✅ บันทึกโมเดล ensemble_model.pkl สำเร็จ!")
