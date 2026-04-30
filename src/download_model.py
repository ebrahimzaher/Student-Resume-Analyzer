import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# تحديد المسار الرئيسي للبروجكت أوتوماتيك
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(BASE_DIR, "models", "Qwen-1.5B")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Starting download for {MODEL_NAME}...")
print(f"Model will be saved exactly to: {SAVE_PATH}")
print("This might take a few minutes depending on your internet speed. Please wait...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(SAVE_PATH)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.save_pretrained(SAVE_PATH)

print(f"✅ Model downloaded and saved successfully to '{SAVE_PATH}'")