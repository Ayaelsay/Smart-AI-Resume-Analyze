import spacy
from transformers import pipeline
import requests

# تحميل النموذج اللغوي من spaCy
nlp = spacy.load("en_core_web_sm")

# عنوان API لتحليل السيرة الذاتية (استبدل بالعنوان الذي أنشأه ngrok)
url = " https://20bc-197-56-115-126.ngrok-free.app/analyze_cv"
# مسار ملف السيرة الذاتية (تأكد أن المسار صحيح)
file_path = r"C:\Users\THINKPAD\Downloads\uj_44917+SOURCE1+SOURCE1.2.pdf"

try:
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        if response.text.strip():
            try:
                data = response.json()
                print("✅ استجابة API:", data)
            except requests.exceptions.JSONDecodeError:
                print("❌ خطأ: الرد ليس بصيغة JSON. الرد الفعلي:", response.text)
        else:
            print("❌ خطأ: الرد من API فارغ.")
    else:
        print(f"❌ فشل الطلب! كود الحالة: {response.status_code}\nالرد: {response.text}")

except FileNotFoundError:
    print(f"❌ الملف {file_path} غير موجود، تأكد من اسم الملف.")

# تحميل نموذج التعرف على الكيانات باستخدام BERT
er_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def analyze_text(text):
    """تحليل النص باستخدام نماذج NLP المجانية."""
    
    # تحليل باستخدام spaCy
    doc = nlp(text)
    print("\n✅ التحليل باستخدام spaCy:")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")
    
    # تحليل باستخدام BERT
    print("\n✅ التحليل باستخدام BERT:")
    entities = er_model(text)
    extracted_entities = {entity['word']: entity.get('entity_group', 'UNKNOWN') for entity in entities}
    print(extracted_entities if extracted_entities else "Not found")

# ✅ جرب التحليل على نص تجريبي
test_text = "John Doe is a Data Scientist with experience in Python and Machine Learning."
analyze_text(test_text)
