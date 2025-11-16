from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# This MUST be the directory where your script saved the files
local_model_dir = "./model" 

print(f"Attempting to load model and tokenizer from {local_model_dir}...")

try:
    # local_files_only=True forces it to load from disk.
    # If it fails, it means files are missing, corrupt, or were never saved.
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_dir, 
        local_files_only=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True 
    )
    
    print("✅ Success! Model and tokenizer loaded from local files.")

    # --- Run a quick functional test ---
    print("\nRunning a quick functional test...")
    prompt = "The capital of India is"
    # Use the model's device (e.g., 'cuda:0')
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
    
    # Generate a few new tokens
    outputs = model.generate(**inputs, max_new_tokens=5)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Prompt: {prompt}")
    print(f"Model Output: {decoded_output}")
    
    if len(decoded_output) > len(prompt):
        print("✅ Functional test passed! Model is generating text.")
    else:
        print("⚠️ Functional test might have failed. Model did not generate new tokens.")

except Exception as e:
    print(f"\n❌ FAILED to load from local files: {e}")
    print("This likely means files are missing, corrupt, or not in the expected './model' directory.")