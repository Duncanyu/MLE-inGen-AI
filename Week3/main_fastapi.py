from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from src.asr.asr_pipeline import get_prompt
from src.llm.openai_llm import gpt_respond
from src.llm.hf_llm import hf_respond
from src.tts.speak import generate_tts
from config import (
    OPENAI_API, OPENAI_MODEL,
    HF_DEVICE, HF_MODEL_NAME, HF_TOKEN,
    LLM_PROVIDER,
    ASR_MODEL, ASR_DEVICE,
    TTS_MODEL, TTS_SPEAKER
)

app = FastAPI()

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    user_text = get_prompt(ASR_MODEL, ASR_DEVICE, audio_bytes)
    print(f"[User]: {user_text}")

    if LLM_PROVIDER == "openai":
        bot_text = gpt_respond(user_text, OPENAI_API, OPENAI_MODEL)
    elif LLM_PROVIDER == "huggingface":
        bot_text = hf_respond(user_text, HF_MODEL_NAME, HF_DEVICE, HF_TOKEN)
    else:
        return {"error": f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}"}

    print(f"[Assistant]: {bot_text}")

    audio_path = generate_tts(bot_text, TTS_MODEL, TTS_SPEAKER)

    return FileResponse(audio_path, media_type="audio/wav")
