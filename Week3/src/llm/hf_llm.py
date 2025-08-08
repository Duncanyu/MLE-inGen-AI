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
            print(f"Loading HF LLM model: '{model_name}'... (will check cache)")

            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map = device,
                torch_dtype = "auto"
            )

            model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

        tokenizer = model_pipeline.tokenizer
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize = False,
            add_generation_prompt = True
        )

        results = model_pipeline(
            formatted_prompt,
            temperature = 0.3,
            max_new_tokens = 200,
            return_full_text = False
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
        return "could not generate a proper response"
