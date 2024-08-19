from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the GPT-J model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    gptj_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Error loading GPT-J model: {e}")

def generate_response(prompt):
    try:
        # Use the pipeline to generate a response
        response = gptj_pipeline(prompt, max_length=100, return_full_text=False)[0]['generated_text']
        return response
    except Exception as e:
        return f"AI Error: Failed to generate response using GPT-J. {str(e)}"

# Example usage
if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = generate_response(prompt)
    print(f"Response: {response}")
