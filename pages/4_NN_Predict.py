import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="วิเคราะห์รีวิว AI - 6704062612049", layout="wide", page_icon="🧠")

st.title("🧠 ระบบวิเคราะห์ความรู้สึกจากรีวิว (Neural Network)")
st.write("---")

# โหลดข้อมูลเพื่อแสดงสถิติ
df_review = pd.read_csv('datasets/review_data.csv')

# ส่วนที่เพิ่ม: ตารางสรุปสถิติรีวิว
st.subheader("📊 สถิติภาพรวมของข้อมูลรีวิว")
col_stats1, col_stats2 = st.columns([1, 2])

with col_stats1:
    # นับจำนวน Positive (1) และ Negative (0)
    stats = df_review['Sentiment'].value_counts().rename({1: 'Positive (บวก)', 0: 'Negative (ลบ)'})
    st.table(stats)

with col_stats2:
    st.write("💡 **วิเคราะห์ข้อมูล:** ชุดข้อมูลประกอบด้วยรีวิวหลากหลายรูปแบบ เพื่อให้ Neural Network เรียนรู้ความแตกต่างของภาษาที่ใช้ชมและตำหนิสินค้า")

st.write("---")

# --- โค้ดส่วนการทำนาย (คงเดิม) ---
@st.cache_resource
def load_and_train():
    # ... (ส่วนฝึกโมเดลเดิมของคุณ) ...
    return None, None # สมมติว่ามี return จริงตามโค้ดเดิม

# (เรียกใช้งานส่วนทำนายตามปกติ)
