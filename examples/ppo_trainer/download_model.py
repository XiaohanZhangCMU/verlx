from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch, sys

# Define model name and local save path
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Replace with any HF model
#MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
LOCAL_PATH = f"/root/model/{MODEL_NAME}"  # Change this to your desired save location

# Step 1: Download the model and tokenizer
print(f"Downloading model: {MODEL_NAME} ...")
config = AutoConfig.from_pretrained(MODEL_NAME, cache_dir=LOCAL_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=LOCAL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=LOCAL_PATH, trust_remote_code=True)


# Save model to disk
print(f"Saving model to: {LOCAL_PATH} ...")
tokenizer.save_pretrained(LOCAL_PATH)
model.save_pretrained(LOCAL_PATH)

sys.exit()
# Step 2: Load model from local storage only
print(f"Reloading model from local directory {LOCAL_PATH} (without internet)...")
local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
local_model = AutoModelForCausalLM.from_pretrained(LOCAL_PATH, local_files_only=True)

print("Model reloaded successfully from local storage!")


