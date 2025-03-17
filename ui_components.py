import streamlit as st
import requests
import pdfplumber
import re

# عنوان API لتحليل السيرة الذاتية
API_URL = "https://3bd2-197-56-115-126.ngrok-free.app/analyze_cv"


# قائمة الشركات ومتطلبات المهارات المطلوبة لها
COMPANIES = {
    "Google": {"skills": ["python", "ml", "ai", "tensorflow"], "min_experience": 3},
    "Microsoft": {"skills": ["c#", "azure", "sql", "dotnet"], "min_experience": 2},
    "Amazon": {"skills": ["aws", "java", "cloud", "devops"], "min_experience": 3},
    "Facebook": {"skills": ["react", "javascript", "frontend", "ui/ux"], "min_experience": 4},
    "Tesla": {"skills": ["automation", "robotics", "ai", "python"], "min_experience": 5}
}

def extract_text_from_pdf(uploaded_file):
    """استخراج النص من ملف PDF"""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def clean_text(text):
    """تنظيف النص من المسافات الزائدة"""
    return re.sub(r'\s+', ' ', text).strip()

def extract_phone_numbers(text):
    """استخراج أرقام الهواتف"""
    phone_pattern = re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
    return list(set(phone_pattern.findall(text)))

def recommend_companies(skills, experience_years):
    """ترشيح الشركات المناسبة بناءً على المهارات وسنوات الخبرة"""
    matched_companies = {}
    for company, data in COMPANIES.items():
        missing_skills = [skill for skill in data["skills"] if skill not in skills]
        experience_needed = data["min_experience"] - experience_years
        if len(missing_skills) < len(data["skills"]):  # لديه بعض المهارات المطلوبة
            matched_companies[company] = {"missing_skills": missing_skills, "experience_needed": max(0, experience_needed)}
    return matched_companies

def upload_and_analyze_pdf():
    """رفع ملف PDF وتحليله"""
    st.subheader("📂 قم برفع سيرتك الذاتية بصيغة PDF")
    uploaded_file = st.file_uploader("اختر ملف PDF", type=["pdf"])
    
    if uploaded_file is not None:
        st.success("✅ تم رفع الملف بنجاح! جاري التحليل...")
        
        # استخراج وتحليل النص
        extracted_text = clean_text(extract_text_from_pdf(uploaded_file))
        phone_numbers = extract_phone_numbers(extracted_text)
        
        # إرسال الملف إلى API
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            data = response.json()
            st.subheader("📊 النتائج المستخرجة:")
            
            # معلومات الاتصال
            st.write("📧 **البريد الإلكتروني:**", ", ".join(data['contact_info']['emails']))
            st.write("📞 **رقم الهاتف:**", ", ".join(phone_numbers) if phone_numbers else "غير متوفر")
            
            # التعليم والخبرات
            experience_years = data.get("experience_years", "غير متوفر")
            try:
                experience_years = int(experience_years)
            except ValueError:
                experience_years = 0  # إذا لم يتم التعرف على عدد سنوات الخبرة
            
            st.write("🎓 **التعليم:**", "\n".join(data["education"]))
            st.write("💼 **سنوات الخبرة:**", experience_years if experience_years else "غير متوفر")
            
            if experience_years < 2:
                st.warning("⚠️ لديك خبرة قليلة، قد تحتاج إلى اكتساب المزيد من الخبرة قبل التقديم على بعض الشركات.")
            
            # المهارات وترشيح الشركات
            skills = data.get("skills", [])
            st.write("🛠 **المهارات المكتشفة:**", ", ".join(skills))
            
            recommended_companies = recommend_companies(skills, experience_years)
            if recommended_companies:
                st.subheader("🏢 الشركات المناسبة لك:")
                for company, details in recommended_companies.items():
                    missing_skills = details["missing_skills"]
                    experience_needed = details["experience_needed"]
                    message = f"✅ {company}"
                    if missing_skills:
                        message += f" (المهارات الناقصة: {', '.join(missing_skills)})"
                    if experience_needed > 0:
                        message += f" | ⚠️ تحتاج إلى {experience_needed} سنوات إضافية من الخبرة."
                    else:
                        message += " | ✅ لديك الخبرة الكافية."
                    st.write(message)
            else:
                st.warning("⚠️ لم يتم العثور على شركات تناسب مهاراتك الحالية.")
            
            # تحليل BERT
            st.write("🤖 **التحليل باستخدام BERT:**")
            st.json(data["bert_analysis"])
            
            # عينة من النص المستخرج
            st.write("📄 **عينة من النص المستخرج:**")
            st.text(extracted_text[:500])
        else:
            st.error("❌ حدث خطأ أثناء تحليل السيرة الذاتية. حاول مرة أخرى.")

# تشغيل الواجهة
st.title("📄 تحليل السيرة الذاتية باستخدام الذكاء الاصطناعي")
upload_and_analyze_pdf()
