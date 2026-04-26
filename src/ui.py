import streamlit as st
import pdfplumber
import re
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', ' ', text)          # remove emails
    text = re.sub(r'\+?\d[\d -]{8,}\d', ' ', text) # remove phones
    text = re.sub(r'http\S+|www\S+', ' ', text)    # remove links
    text = re.sub(r'[^a-zA-Z ]', ' ', text)        # remove symbols
    text = re.sub(r'\s+', ' ', text).strip()       # normalize spaces
    return text

def trim_text(text, max_words=300):
    return " ".join(text.split()[:max_words])

def create_prompt(cv, role):
    return f"""
You are an expert HR system.

Evaluate this CV against ANY given job role.

CV:
{cv}

Target Job Role:
{role}

Tasks:
1. Is this CV a GOOD or BAD fit?
2. Give a score from 0 to 10
3. Give 3 improvement suggestions specific to this role

STRICT:
Return ONLY JSON:
{{
  "label": "good" or "bad",
  "score": number,
  "suggestions": ["...", "...", "..."]
}}
"""

def safe_parse(text):
    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}") + 1
        try:
            return json.loads(text[start:end])
        except:
            return None

def analyze_cv(cv, role):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": create_prompt(cv, role)}]
    )

    parsed = safe_parse(response.choices[0].message.content)

    if parsed:
        return parsed
    else:
        return {"label": "unknown", "score": 0, "suggestions": ["Parsing error"]}

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI Resume Analyzer")
st.write("Upload your CV and evaluate it against ANY job role")

default_roles = [
    "Data Scientist",
    "Machine Learning Engineer",
    "Backend Developer",
    "Frontend Developer",
    "Mobile App Developer"
]

selected = st.selectbox("Choose a common role (or select Custom)", ["Custom"] + default_roles)

if selected == "Custom":
    role = st.text_input("Enter Target Job Role")
else:
    role = selected

# Validation
if not role or role.strip() == "":
    st.warning("⚠️ Please enter a job role to continue")

uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF uploaded successfully ✅")

    if st.checkbox("Show extracted text"):
        raw_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text", raw_text, height=200)

    if st.button("Analyze CV"):

        if not role or role.strip() == "":
            st.error("Please enter a job role first")
        else:
            with st.spinner("Analyzing your CV..."):

                raw_text = extract_text_from_pdf(uploaded_file)
                cleaned = clean_text(raw_text)
                final_text = trim_text(cleaned)

                result = analyze_cv(final_text, role)

            st.subheader("📊 Result")

            if result["label"] == "good":
                st.success(f"GOOD FIT ✅ (Score: {result['score']}/10)")
            elif result["label"] == "bad":
                st.error(f"BAD FIT ❌ (Score: {result['score']}/10)")
            else:
                st.warning("Unknown result")

            try:
                st.progress(result["score"] / 10)
            except:
                pass

            st.subheader("💡 Suggestions")

            for s in result.get("suggestions", []):
                st.write(f"- {s}")