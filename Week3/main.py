import os

from src.llm.openai_llm import gpt_respond
from src.llm.hf_llm import hf_respond
from src.asr.asr_pipeline import get_prompt
from src.tts.speak import generate_tts

from config import OPENAI_API, OPENAI_MODEL, HF_DEVICE, HF_MODEL_NAME, HF_TOKEN
from config import LLM_PROVIDER, ASR_DEVICE, ASR_MODEL, TTS_MODEL, TTS_SPEAKER

if LLM_PROVIDER == "openai":
    while True:
        audio = generate_tts(gpt_respond(get_prompt(ASR_MODEL, ASR_DEVICE), OPENAI_API, OPENAI_MODEL), TTS_MODEL, TTS_SPEAKER)
        
        os.system(f"afplay {audio}") 
        os.remove(audio)
elif LLM_PROVIDER == "huggingface":
    while True:
        audio = generate_tts(hf_respond(get_prompt(ASR_MODEL, ASR_DEVICE), HF_MODEL_NAME, HF_DEVICE, HF_TOKEN), TTS_MODEL, TTS_SPEAKER)
        
        os.system(f"afplay {audio}")
        os.remove(audio)

# print(hf_respond("Hello", HF_MODEL_NAME, HF_DEVICE, HF_TOKEN), TTS_MODEL, TTS_SPEAKER)