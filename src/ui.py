import streamlit as st
import PyPDF2
import re
import os
import json
import torch
from transformers import pipeline


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = os.path.join(BASE_DIR, "models", "Qwen-1.5B")

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

hf_pipeline = load_model()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\+?\d[\d -]{8,}\d', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text

def create_prompt(cv_context, category):
    return f"""Analyze the provided CV against the position of {category}.
Strictly structure your response using XML tags exactly as follows:

<Detailed_Analysis>
Write a detailed 3-4 sentence analysis. Mention specifically what is strong in the CV and what is missing relative to the {category} role context found in <cv_text>.
</Detailed_Analysis>

<Actionable_Tips>
Provide exactly 3 specific, actionable advice bullet points for the candidate to improve this CV for the {category} role. Focus on practical changes they can make.
</Actionable_Tips>

<Final_Classification>
Return ONLY a valid JSON object block: {{"label": "good" or "bad"}}
</Final_Classification>

Target Role: {category}

<cv_text>
{cv_context}
</cv_text>
"""

def parse_sequential_response(text):
    result = {
        "label": "unknown",
        "analysis": "No analysis generated. (Output likely truncated)",
        "tips": "No specific tips provided. (Output likely truncated)"
    }
    
    json_start = text.rfind("{")
    json_end = text.rfind("}") + 1
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            json_str = text[json_start:json_end]
            json_data = json.loads(json_str)
            result["label"] = json_data.get("label", "unknown").lower()
        except:
            pass

    def extract_between_tags_robustly(full_text, open_tag):
        start_idx = full_text.find(open_tag)
        if start_idx == -1:
            return None
        
        start_idx += len(open_tag)
        close_tag = open_tag.replace("<", "</")
        end_idx = full_text.find(close_tag)
        
        if end_idx != -1:
            return full_text[start_idx:end_idx].strip()
        
        return full_text[start_idx:].strip()
    
    analysis_raw = extract_between_tags_robustly(text, "<Detailed_Analysis>")
    if analysis_raw:
        result["analysis"] = analysis_raw
        result["analysis"] = result["analysis"].split("<Actionable_Tips>")[0].strip()

    tips_raw = extract_between_tags_robustly(text, "<Actionable_Tips>")
    if tips_raw:
        tips_cleaned = tips_raw.split("<Final_Classification>")[0].split("{")[0].strip()
        
        lines = tips_cleaned.split('\n')
        if lines:
             last_line = lines[-1].strip()
             if re.search(r'\s(or|and|the|a|to|with|in)\s*$', last_line) or last_line.endswith('.') == False:
                 if len(lines) > 1:
                     tips_cleaned = "\n".join(lines[:-1]).strip()
        
        formatted_lines = []
        for line in tips_cleaned.split('\n'):
            line = line.strip()
            if not line: continue
            clean_line = re.sub(r'^\s*[\d•*-]+\.?\s*', '', line)
            formatted_lines.append(f"• {clean_line}")
        
        if formatted_lines:
            result["tips"] = "\n".join(formatted_lines[:3])

    return result

st.set_page_config(page_title="Match-Your-CV AI", page_icon="📄", layout="centered")

st.title("📄 AI Resume Analyzer")
st.markdown("Upload a PDF resume and let Qwen-1.5B evaluate its match for your target role.")

target_role = st.text_input("Enter Target Role (e.g., Data Scientist, HR, Advocate):", "Data Scientist")
uploaded_file = st.file_uploader("Upload CV (PDF format)", type=["pdf"])

if st.button("🚀 Analyze CV"):
    if not uploaded_file:
        st.warning("Please upload a PDF file first!")
    elif not target_role:
        st.warning("Please enter a target role!")
    else:
        with st.spinner("Analyzing your CV with AI... Please wait."):
            raw_text = extract_text_from_pdf(uploaded_file)
            cleaned_text = clean_text(raw_text)
            
            final_text = " ".join(cleaned_text.split()[:400]) 
            
            prompt = create_prompt(final_text, target_role)
            messages = [
                {"role": "system", "content": "You are an expert HR system."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                result = hf_pipeline(
                    messages, 
                    max_new_tokens=600, 
                    max_length=None,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True
                )
                
                generated_text = result[0]['generated_text'][-1]['content'].strip()
                parsed_result = parse_sequential_response(generated_text)
                
                st.divider()
                st.subheader("📊 Evaluation Result")
                
                label = parsed_result["label"]
                
                if label == "good":
                    st.success(f"**Match Status:** {label.upper()} ✅")
                elif label == "bad":
                    st.error(f"**Match Status:** {label.upper()} ❌")
                else:
                    st.warning(f"**Match Status:** {label.upper()} ⚠️")
                
                st.subheader("💡 Detailed AI Feedback")
                st.markdown("### 🔍 Analysis:")
                st.write(parsed_result["analysis"])
                
                st.markdown("### ✏️ Specific Tips for Improvement:")
                st.markdown(parsed_result["tips"])
                
                with st.expander("🛠️ Show Raw AI Chain-of-Thought (For Debugging)"):
                    st.code(generated_text, language="xml")
                    
            except Exception as e:
                st.error(f"An error occurred during inference: {e}")
