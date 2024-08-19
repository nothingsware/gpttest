from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

def generate_response(prompt):
    model_name = "EleutherAI/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
        response = generate_response(prompt)
        print(response)
    else:
        print("No prompt provided.")
