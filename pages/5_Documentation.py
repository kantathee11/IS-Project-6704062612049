import streamlit as st

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Project Documentation", layout="wide", page_icon="📄")

def show_doc():
    st.title("📄 เอกสารประกอบโครงการ")
    st.write("รายละเอียดแนวทางการพัฒนาโมเดล ขั้นตอนการเตรียมข้อมูล และทฤษฎีที่เกี่ยวข้อง")
    st.divider()

    try:
        # อ่านไฟล์ DOCUMENTATION.md
        with open("DOCUMENTATION.md", "r", encoding="utf-8") as f:
            doc_content = f.read()
        
        # แสดงเนื้อหา Markdown
        st.markdown(doc_content)
        
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์ DOCUMENTATION.md ใน Repository")
        st.info("กรุณาตรวจสอบว่าคุณได้สร้างไฟล์ชื่อ DOCUMENTATION.md ไว้ที่โฟลเดอร์หลักแล้วหรือยัง")

    st.write("---")
    st.caption("รหัสนักศึกษา: 6704062612049 | IS Project 2026")

if __name__ == "__main__":
    show_doc()

# เพิ่มปุ่มดาวน์โหลดไฟล์ PDF (ถ้ามีไฟล์ Document.pdf อยู่ในเครื่อง/GitHub)
    try:
        with open("Document.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        st.download_button(label="📥 ดาวน์โหลดเอกสารฉบับเต็ม (PDF)",
                           data=PDFbyte,
                           file_name="Project_Documentation_6704062612049.pdf",
                           mime='application/octet-stream')
    except:
        pass
