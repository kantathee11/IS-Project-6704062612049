import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# 1. โหลดข้อมูล
print("กำลังโหลดข้อมูล review_data.csv...")
df = pd.read_csv('datasets/review_data.csv')

# 2. ทำความสะอาดข้อความ (Text Preprocessing)
# ลบ HTML Tags และปรับเป็นตัวพิมพ์เล็ก
df['Review_Text'] = df['Review_Text'].str.replace(r'<[^>]*>', '', regex=True).str.lower().str.strip()

# 3. แปลงข้อความเป็นตัวเลขด้วย TF-IDF
print("กำลังแปลงข้อความด้วย TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Review_Text'])
y = df['Sentiment']

# 4. สร้างโมเดล Neural Network (MLP)
print("กำลังฝึกสอน Neural Network (MLP)...")
model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
model.fit(X, y)

# 5. บันทึกโมเดลและ Vectorizer ลงโฟลเดอร์ models
if not os.path.exists('models'):
    os.makedirs('models')

# บันทึกทั้งตัวโมเดลและตัวแปลงข้อความ (สำคัญมากต้องใช้คู่กัน)
with open('models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ บันทึกโมเดลและ Vectorizer สำเร็จ!")
