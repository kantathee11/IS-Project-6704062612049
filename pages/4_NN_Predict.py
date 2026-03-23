import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# 1. การตั้งค่าหน้าเว็บให้ดูพรีเมียม
st.set_page_config(page_title="AI Sentiment Analysis", layout="wide", page_icon="🧠")

st.title("🧠 AI Sentiment Analysis (Neural Network)")
st.subheader("ระบบวิเคราะห์ความรู้สึกจากข้อความด้วยโครงข่ายประสาทเทียม")
st.write("---")

# 2. เบื้องหลังการทำงาน (Processing)
# โหลดข้อมูลและเตรียม Vectorizer (Tfidf ช่วยให้ AI เข้าใจน้ำหนักคำได้ดีขึ้น)
@st.cache_resource # ใช้ cache เพื่อให้เว็บรันเร็วขึ้น ไม่ต้องฝึกโมเดลใหม่ทุกครั้งที่กด
def train_nn_model():
    df = pd.read_csv('datasets/review_data.csv')
    # Clean ข้อมูลเบื้องต้น
    df['Review_Text'] = df['Review_Text'].str.replace(r'<[^>]*>', '', regex=True).str.lower().str.strip()
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Review_Text'])
    y = df['Sentiment']
    
    # ล็อก random_state=42 เพื่อให้ผลลัพธ์ "นิ่ง" ไม่สลับไปมา
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = train_nn_model()

# 3. ส่วนการจัดวางหน้าจอ (Layout)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📝 ระบุข้อความรีวิว")
    st.write("ป้อนข้อความภาษาอังกฤษที่คุณต้องการให้ AI วิเคราะห์ความรู้สึก (Sentiment)")
    
    # ช่องรับข้อมูลแบบ Text Area
    user_input = st.text_area("กรอกรีวิวของคุณที่นี่:", 
                             placeholder="เช่น The product is amazing and very useful...",
                             height=150)
    
    analyze_btn = st.button("🚀 เริ่มวิเคราะห์ความรู้สึก", use_container_width=True)

with col2:
    st.markdown("### 📊 ผลการวิเคราะห์")
    if analyze_btn:
        if user_input.strip() != "":
            with st.spinner('AI กำลังถอดรหัสข้อความ...'):
                # แปลงข้อความเป็นตัวเลขและทำนาย
                input_vector = vectorizer.transform([user_input.lower()])
                prediction = model.predict(input_vector)
                
                st.write("คำนวณผลลัพธ์สำเร็จ!")
                if prediction[0] == 1:
                    st.success("### 😊 ผลลัพธ์: เชิงบวก (Positive)")
                    st.write("AI วิเคราะห์ว่าข้อความนี้แสดงถึง **ความพึงพอใจ**")
                    st.balloons()
                else:
                    st.error("### 😞 ผลลัพธ์: เชิงลบ (Negative)")
                    st.write("AI วิเคราะห์ว่าข้อความนี้แสดงถึง **ความไม่พึงพอใจ**")
        else:
            st.warning("กรุณากรอกข้อความก่อนกดวิเคราะห์ครับ")
    else:
        st.info("รอรับข้อมูลเพื่อทำการวิเคราะห์...")

# 4. ส่วนแสดงทฤษฎีเล็กน้อย (Footer)
st.write("---")
st.markdown("""
**💡 ข้อมูลเชิงเทคนิค:**
โมเดลนี้ใช้โครงสร้าง **Multi-Layer Perceptron (MLP)** ที่มีเลเยอร์ซ่อน 2 ชั้น (32 และ 16 Neurons) 
ทำงานร่วมกับเทคนิค **TF-IDF** เพื่อสกัดความสำคัญของคำศัพท์ (Keywords) จากข้อมูลรีวิวจำลอง
""")
