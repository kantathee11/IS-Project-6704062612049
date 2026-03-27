# [cite_start]เอกสารประกอบโครงการ (Project Documentation) [cite: 1]

[cite_start]**ชื่อโครงการ:** ระบบพยากรณ์ความเสี่ยงสุขภาพและวิเคราะห์ความรู้สึกด้วย AI [cite: 2]  
[cite_start]**รหัสนักศึกษา:** 6704062612049 [cite: 3]

---

## [cite_start]1. แนวทางการพัฒนาโมเดล (Model Development Approach) [cite: 4]

### [cite_start]การเตรียมข้อมูล (Data Preparation) [cite: 5]
* [cite_start]**ที่มาของข้อมูล:** โปรเจกต์นี้ใช้ข้อมูลที่ผ่านการสังเคราะห์ให้มีความใกล้เคียงกับสถานการณ์จริง แบ่งเป็น 2 ชุดข้อมูลหลัก ได้แก่ `heart_data.csv` (ข้อมูลสุขภาพพนักงาน) และ `review_data.csv` (ข้อมูลรีวิวภาษาอังกฤษ) [cite: 6]
* [cite_start]**การจัดการข้อมูล:** มีการตรวจสอบความถูกต้องและทำความสะอาดข้อมูล (Data Cleansing) โดยการลบข้อมูลซ้ำ (Drop Duplicates) และจัดการกับค่าว่าง (Dropna) เพื่อให้ Dataset มีคุณภาพสูงสุดก่อนนำไปสอนโมเดล [cite: 7]
* [cite_start]**การสร้างสถานการณ์:** จำลองข้อมูลให้มีความหลากหลาย ทั้งช่วงอายุ (Age), ระดับคอเลสเตอรอล (Cholesterol) และระดับความเครียด (Stress Level) เพื่อให้โมเดลเรียนรู้รูปแบบความเสี่ยงสุขภาพได้ครอบคลุม [cite: 8]

### [cite_start]ทฤษฎีของอัลกอริทึม (Algorithm Theory) [cite: 9]
[cite_start]โปรเจกต์นี้เลือกใช้เทคนิคการประมวลผล 2 รูปแบบหลัก: [cite: 10]

**1. [cite_start]Ensemble Learning (สำหรับการพยากรณ์สุขภาพ):** [cite: 11]
[cite_start]ใช้เทคนิค **Voting Classifier** เพื่อรวมพลังของ 3 โมเดล[cite: 11]:
* [cite_start]**Random Forest:** ใช้หลักการ Bagging สร้างต้นไม้ตัดสินใจหลายต้นเพื่อลดความแปรปรวน (Variance) [cite: 13]
* **Logistic Regression:** ใช้การหาความสัมพันธ์เชิงเส้นเพื่อระบุโอกาสเกิดความเสี่ยงแบบเป็นเหตุเป็นผล [cite: 14]
* [cite_start]**Support Vector Machine (SVC):** ใช้การสร้างขอบเขต (Margin) ที่กว้างที่สุดเพื่อแยกกลุ่มข้อมูลสุขภาพ [cite: 15]

**2. [cite_start]Neural Network (สำหรับการวิเคราะห์ความรู้สึก):** [cite: 16]
[cite_start]ใช้โครงสร้าง **Multi-layer Perceptron (MLP)** สำหรับงาน Sentiment Analysis[cite: 16]:
* [cite_start]**MLP Classifier:** จำลองการทำงานของระบบประสาทมนุษย์ มี Hidden Layers ในการเรียนรู้ความสัมพันธ์ที่ซับซ้อน [cite: 17]
* **TF-IDF Vectorizer:** แปลงข้อความ (Text) ให้เป็นตัวเลขเชิงสถิติ เพื่อให้ Neural Network ประมวลผลความหมายได้ [cite: 18]

---

## 2. ขั้นตอนการพัฒนาโมเดล (Development Process) [cite: 19]

* [cite_start]**Data Cleaning & Integration:** นำ Dataset เข้าสู่กระบวนการ Preprocessing ลบอักขระพิเศษในรีวิว และปรับสเกลตัวเลขข้อมูลสุขภาพให้พร้อมใช้งาน [cite: 21]
* [cite_start]**Feature Engineering:** ในส่วนของ NN มีการแปลงข้อความเป็นค่าน้ำหนัก TF-IDF และในส่วนของ ML มีการนำตัวแปร Age, Cholesterol และ Stress มาใช้เป็น Feature หลักในการทำนาย [cite: 21]
* **Data Splitting:** แบ่งข้อมูลเพื่อใช้สำหรับการ Training และการประเมินผล เพื่อตรวจสอบความแม่นยำ (Accuracy) [cite: 22]
* [cite_start]**Model Training & Ensemble:** ฝึกสอนโมเดลทั้งส่วนของ Health และ Sentiment พร้อมกันบนระบบ Streamlit [cite: 23]
* [cite_start]**Model Export:** บันทึกโมเดลที่ฝึกเสร็จแล้วลงในไฟล์ `.pkl` (เช่น `ensemble_model.pkl`) ไว้ในโฟลเดอร์ `models/` เพื่อให้นำกลับมาใช้งานได้ทันที [cite: 24, 25]

---

## [cite_start]3. แหล่งอ้างอิงข้อมูล (References) [cite: 26]
* **Scikit-learn Documentation:** ข้อมูลด้าน Algorithm (Voting, MLP, TF-IDF) [cite: 27]
* [cite_start]**Streamlit Documentation:** การพัฒนาส่วน Web Interface [cite: 28]
* [cite_start]**Kaggle / UCI Machine Learning Repository:** แหล่งอ้างอิงชุดข้อมูลจำลอง [cite: 29]
