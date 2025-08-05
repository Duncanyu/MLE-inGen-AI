from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_pipeline = None

def hf_respond(prompt: str, model_name: str, device: str = "auto", hf_token: str = None):
    global model_pipeline
    
    print("Thinking...")
    try:
        if hf_token:
            login(token = hf_token)
            
        if model_pipeline is None:
            print("Loading HF LLM model: '{model_name}... (will check cache)")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = hf_token)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model_pipeline = pipeline("text-generation", model = model, tokenizer = tokenizer)

        results = model_pipeline(prompt, temperature = 0.3)
        if "generated_text" in results[0]:
            return results[0]["generated_text"]
        elif "text" in results[0]:
            return results[0]["text"]
        else:
            raise KeyError("format is weird for this hf model")


    except Exception as e:
        print(f"Error: {e}")
        return "Could not generate a proper response"