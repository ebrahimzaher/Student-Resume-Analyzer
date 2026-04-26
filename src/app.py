import pandas as pd
import re
import os
import time
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INPUT_FILE = "data/input/resume_dataset.csv"
OUTPUT_FILE = "data/output/final_labeled_dataset.csv"
CHECKPOINT_FILE = "data/checkpoint/checkpoint.csv"

CATEGORY_COLUMN = "Category"

df = pd.read_csv(INPUT_FILE)

TEXT_COLUMN = "Resume"

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

def trim_text(text, max_words=300):
    return " ".join(text.split()[:max_words])

df["final_text"] = df["cleaned"].apply(trim_text)

def create_prompt(cv, category):
    return f"""
You are an expert HR system.

CV:
{cv}

Target Job Category:
{category}

Tasks:
1. Evaluate if this CV is a GOOD or BAD fit
2. Give a score from 0 to 10
3. Give 1 actionable improvement suggestion

STRICT:
Return ONLY JSON:
{{
  "label": "good" or "bad",
  "score": number,
  "suggestion": "text"
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

def analyze(cv, category, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-safeguard-20b",
                messages=[{"role": "user", "content": create_prompt(cv, category)}]
            )

            parsed = safe_parse(response.choices[0].message.content)

            if parsed:
                return parsed

        except Exception as e:
            print(f"Retry {attempt+1}: {e}")
            time.sleep(2)

    return {"label": "unknown", "score": 0, "suggestion": "failed"}

if os.path.exists(CHECKPOINT_FILE):
    print("Loading checkpoint...")
    df_existing = pd.read_csv(CHECKPOINT_FILE)
else:
    df_existing = pd.DataFrame()

processed_texts = set(df_existing.get("final_text", []))

cache = {}

def analyze_cached(cv, category):
    key = cv + category

    if key in cache:
        return cache[key]

    result = analyze(cv, category)
    cache[key] = result
    return result

batch_size = 50

results = []

for start in range(0, len(df), batch_size):
    batch = df.iloc[start:start+batch_size]

    print(f"\nProcessing batch {start} → {start+len(batch)}")

    for i, row in batch.iterrows():

        if row["final_text"] in processed_texts:
            continue

        result = analyze_cached(row["final_text"], row[CATEGORY_COLUMN])

        row_result = row.to_dict()
        row_result["label"] = result.get("label", "unknown")
        row_result["score"] = result.get("score", 0)
        
        row_result["suggestion"] = result.get("suggestion", "") 

        results.append(row_result)

        print(f"Processed index {i}")

        time.sleep(1)

    temp_df = pd.DataFrame(results)
    combined = pd.concat([df_existing, temp_df], ignore_index=True)
    combined.to_csv(CHECKPOINT_FILE, index=False)

    print("Checkpoint saved ✅")

final_df = pd.read_csv(CHECKPOINT_FILE)
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n🎉 Pipeline completed successfully!")