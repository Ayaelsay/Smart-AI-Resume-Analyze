import re
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
import spacy
from transformers import pipeline
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)  # تشغيل التطبيق مع ngrok

# تحميل النموذج اللغوي
nlp = spacy.load("en_core_web_sm")

# تحميل نموذج التعرف على الكيانات باستخدام Hugging Face
er_model = pipeline("ner", model="dslim/bert-base-NER", device=-1, aggregation_strategy="simple")

# قائمة المهارات التقنية الشائعة
SKILLS = {
    "python", "java", "javascript", "react", "node.js", "sql", "html", "css",
    "aws", "docker", "kubernetes", "git", "machine learning", "ai", "data science",
    "deep learning", "nlp", "flask", "django", "tensorflow", "pytorch"
}

# قائمة الشركات ومتطلبات المهارات
COMPANIES = {
    "Facebook": {"required_skills": {"Frontend", "UI/UX", "TypeScript"}},
    "Google": {"required_skills": {"Redux", "Web Performance Optimization"}},
    "Amazon": {"required_skills": {"Node.js", "Express.js", "AWS"}}
}

def extract_text_from_pdf(pdf_file):
    """استخراج النصوص من ملف PDF."""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text("text") for page in pdf_document).strip()
    return text if text else None

phone_pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

def extract_contact_info(text):
    """استخراج البريد الإلكتروني ورقم الهاتف."""
    emails = re.findall(email_pattern, text)
    phone_numbers = re.findall(phone_pattern, text)
    return {
        "emails": emails if emails else ["Not found"],
        "phone_numbers": phone_numbers if phone_numbers else ["Not found"]
    }

def extract_skills(text):
    """استخراج المهارات من النص."""
    doc = nlp(text.lower())
    found_skills = {token.text for token in doc if token.text in SKILLS}
    return list(found_skills) or ["Not found"]

def extract_experience(text):
    """تحليل سنوات الخبرة من النص."""
    experience_pattern = r"(\d{1,2})\s*(?:years?|yrs?|year|experience|of experience)"
    matches = re.findall(experience_pattern, text, re.IGNORECASE)
    years_of_experience = [int(match) for match in matches if match.isdigit()]
    return max(years_of_experience, default="Not found")

def extract_education(text):
    """استخراج الدرجات الأكاديمية فقط."""
    education_keywords = ["bachelor", "master", "phd", "degree", "university", "college", "education"]
    doc = nlp(text.lower())
    education_sentences = [
        sent.text for sent in doc.sents 
        if any(word in sent.text for word in education_keywords) and len(sent.text) < 120
    ]
    return education_sentences if education_sentences else ["Not found"]

def analyze_with_bert(text):
    """تحليل النص باستخدام نموذج BERT."""
    entities = er_model(text[:1000])
    extracted_entities = {entity.get('word', 'UNKNOWN'): entity.get('entity_group', 'UNKNOWN') for entity in entities}
    return extracted_entities if extracted_entities else "Not found"

def recommend_jobs(skills):
    """ترشيح الوظائف بناءً على المهارات المكتشفة."""
    recommended_jobs = []
    cv_skills = set(skills)
    for company, details in COMPANIES.items():
        missing_skills = details["required_skills"] - cv_skills
        if len(missing_skills) < len(details["required_skills"]):
            recommended_jobs.append({"company": company, "missing_skills": list(missing_skills)})
    return recommended_jobs if recommended_jobs else "No suitable jobs found"

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ API is running!"})

@app.route("/analyze_cv", methods=["POST"])
def analyze_cv():
    """تحليل السيرة الذاتية باستخدام نماذج NLP المجانية."""
    if "file" not in request.files:
        return jsonify({"error": "❌ No file uploaded"}), 400

    file = request.files["file"]
    text = extract_text_from_pdf(file)
    if not text:
        return jsonify({"error": "❌ Could not extract text from PDF"}), 400

    contact_info = extract_contact_info(text)
    skills = extract_skills(text)
    experience = extract_experience(text)
    education = extract_education(text)
    bert_analysis = analyze_with_bert(text)
    job_recommendations = recommend_jobs(skills)

    return jsonify({
        "message": f"✅ Processed file: {file.filename}",
        "contact_info": contact_info,
        "skills": skills,
        "experience_years": experience,
        "education": education,
        "bert_analysis": bert_analysis,
        "job_recommendations": job_recommendations,
        "extracted_text_sample": text[:500]
    })

if __name__ == "__main__":
    app.run()
