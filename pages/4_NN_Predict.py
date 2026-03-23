import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="NN Sentiment Analysis", layout="wide", page_icon="🧠")

st.title("🧠 ระบบวิเคราะห์ความรู้สึกจากรีวิว (Neural Network)")
st.write("---")

# 2. ส่วนข้อมูลสถิติ (Statistics Section)
st.subheader("📊 สถิติภาพรวมของข้อมูลรีวิว")

try:
    # โหลดไฟล์ข้อมูล
    df_review = pd.read_csv('datasets/review_data.csv')
    
    # นับจำนวน Positive (1) และ Negative (0)
    pos_count = len(df_review[df_review['Sentiment'] == 1])
    neg_count = len(df_review[df_review['Sentiment'] == 0])
    
    # สร้าง DataFrame ใหม่สำหรับแสดงตารางให้สวยงาม
    stats_data = {
        "ประเภทรีวิว": ["Positive (เชิงบวก)", "Negative (เชิงลบ)", "รวมทั้งหมด"],
        "จำนวน (รายการ)": [f"{pos_count} รีวิว", f"{neg_count} รีวิว", f"{len(df_review)} รีวิว"]
    }
    df_stats = pd.DataFrame(stats_data)

    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.table(df_stats) # แสดงตารางสรุป
    with col_t2:
        st.info("💡 ข้อมูลชุดนี้ใช้สอนให้ Neural Network จดจำลักษณะคำศัพท์ที่บ่งบอกถึงความพึงพอใจและไม่พึงพอใจ")

except Exception as e:
    st.error("ไม่สามารถโหลดไฟล์ข้อมูลได้ กรุณาตรวจสอบ datasets/review_data.csv")

st.write("---")

# 3. ส่วนฝึกสอนโมเดล (Training Section)
@st.cache_resource
def train_model():
    df = pd.read_csv('datasets/review_data.csv')
    # ลบ HTML Tags และปรับเป็นตัวพิมพ์เล็ก
    df['Review_Text'] = df['Review_Text'].str.replace(r'<[^>]*>', '', regex=True).str.lower().str.strip()
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Review_Text'])
    y = df['Sentiment']
    
    # สร้างโมเดลแบบ MLP
    model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = train_model()

# 4. ส่วนช่องกรอกข้อมูล (Input Section)
with st.container(border=True):
    st.subheader("📝 ทดสอบพิมพ์รีวิวของคุณ")
    user_text = st.text_area("กรอกข้อความภาษาอังกฤษที่นี่:", placeholder="เช่น The service was great, I love it!")
    
    if st.button("🚀 วิเคราะห์ผล", use_container_width=True):
        if user_text.strip() != "":
            with st.spinner('กำลังประมวลผล...'):
                input_vector = vectorizer.transform([user_text.lower()])
                prediction = model.predict(input_vector)
                
                st.write("---")
                if prediction[0] == 1:
                    st.success("### 😊 ผลลัพธ์: เชิงบวก (Positive)")
                    st.balloons()
                else:
                    st.error("### 😞 ผลลัพธ์: เชิงลบ (Negative)")
        else:
            st.warning("กรุณาพิมพ์ข้อความก่อนกดวิเคราะห์")

st.write("---")
st.info("💡 หมายเหตุ: ใช้ Neural Network แบบ Multi-layer Perceptron (MLP) ในการประมวลผล")
