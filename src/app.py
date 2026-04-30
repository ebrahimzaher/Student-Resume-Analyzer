import pandas as pd
import re
import os
import json
import numpy as np
import torch
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(BASE_DIR, "data", "input", "resume_dataset.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "output", "final_labeled_dataset.csv")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "data", "checkpoint", "checkpoint.csv")
MODEL_NAME = os.path.join(BASE_DIR, "models", "Qwen-1.5B")
CATEGORY_COLUMN = "Category"
TEXT_COLUMN = "Resume"

hf_pipeline = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model Loaded Successfully!")

df = pd.read_csv(INPUT_FILE)

if TEXT_COLUMN not in df.columns:
    raise Exception(f"Column '{TEXT_COLUMN}' not found! Available columns: {df.columns}")

print(f"Using column: {TEXT_COLUMN}")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\+?\d[\d -]{8,}\d', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["cleaned"] = df[TEXT_COLUMN].apply(clean_text)

df = df[df["cleaned"].str.len() > 50]
df = df.drop_duplicates(subset=["cleaned"])

def trim_text(text, max_words=200):
    return " ".join(text.split()[:max_words])

df["final_text"] = df["cleaned"].apply(trim_text)

def extract_and_select_features(text_series, top_k=15):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(text_series)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    selected_features_list = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(i).toarray()[0]
        top_indices = row.argsort()[-top_k:][::-1]
        selected_features = feature_names[top_indices]
        selected_features_list.append(", ".join(selected_features))
        
    return selected_features_list

df["selected_features"] = extract_and_select_features(df["final_text"])

def create_prompt(features, cv_context, category):
    return f"""
Evaluate if the CV is a good match for the target role.
Return ONLY a valid JSON object. No explanation, no intro.

Target Role: {category}

<cv_text>
{cv_context}
</cv_text>

Evaluation Rules:
- "good": The CV shows relevant skills, experience, or education for the target role.
- "bad": The CV is completely irrelevant or lacks the core requirements for the target role.
- "suggestion": MUST contain one explaining exactly why it is good or bad based ONLY on <cv_text>.

Format exactly like this:
{{"label": "good" or "bad", "suggestion": "write your short explanation here"}}
"""

def safe_parse(text):
    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except:
                return None
    return None


def analyze_local(features, cv_context, category):
    prompt = create_prompt(features, cv_context, category)
    
    messages = [
        {"role": "system", "content": "You are an expert HR system."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        result = hf_pipeline(
            messages, 
            max_new_tokens=150,
            max_length=None,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )
        
        generated_text = result[0]['generated_text'][-1]['content'].strip()
        parsed = safe_parse(generated_text)
        
        if parsed:
            return parsed
            
    except Exception as e:
        print(f"Inference Error: {e}")
        
    return {"label": "unknown", "score": 0, "suggestion": "Failed to parse"}

if os.path.exists(CHECKPOINT_FILE):
    print("Loading checkpoint...")
    df_existing = pd.read_csv(CHECKPOINT_FILE)
else:
    df_existing = pd.DataFrame()

processed_texts = set(df_existing.get("final_text", []))

batch_size = 10
results = []

print("Starting Evaluation Pipeline...")

for start in range(0, len(df), batch_size):
    batch = df.iloc[start:start+batch_size]
    print(f"\nProcessing batch {start} → {start+len(batch)}")

    for i, row in batch.iterrows():
        if row["final_text"] in processed_texts:
            continue

        result = analyze_local(
            features=row["selected_features"], 
            cv_context=row["final_text"], 
            category=row[CATEGORY_COLUMN]
        )

        row_result = row.to_dict()
        row_result["label"] = result.get("label", "unknown")
        row_result["suggestion"] = result.get("suggestion", "") 

        results.append(row_result)
        print(f"Processed index {i}")

    if results:
        temp_df = pd.DataFrame(results)
        combined = pd.concat([df_existing, temp_df], ignore_index=True)
        combined.to_csv(CHECKPOINT_FILE, index=False)
        df_existing = combined
        print("Checkpoint saved ✅")
        results = [] 

final_df = pd.read_csv(CHECKPOINT_FILE)
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n🎉 Mission completed successfully!")
