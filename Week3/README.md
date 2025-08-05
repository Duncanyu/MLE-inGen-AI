# Voice Research Agent

A lightweight and modular voice interaction agent that enables real-time speech-to-speech conversations. The system combines **Whisper-based ASR**, customizable **LLMs (OpenAI & Hugging Face)**, and **Coqui TTS** for flexible local or API-powered backends.  
It’s designed to be minimal, configurable, and cross-platform, with the ability to easily swap models and providers for experimentation or production use.  

---

## Getting Started

### 1. Download or Clone

Clone the repository:  
```bash
git clone https://github.com/<your-username>/voice-research-agent.git
cd voice-research-agent
```

Or download the ZIP from GitHub and extract it into your preferred directory.

### 2. Set up a Virtual Environment

**Mac/Linux:**  
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**  
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install `espeak-ng` (for TTS)

**Mac (Homebrew):**  
```bash
brew install espeak-ng
```

**Linux (Debian/Ubuntu):**  
```bash
sudo apt-get update
sudo apt-get install espeak-ng
```

**Windows:**  
Download and install `espeak-ng` from the [official releases](https://github.com/espeak-ng/espeak-ng/releases).  
Make sure to add it to your `PATH` during installation.

### 5. Configure

Duplicate the template configuration file and rename it:  
```bash
cp config_template.py config.py
```

Then open `config.py` and edit the settings (e.g., API keys, model names, devices).

### 6. Run the Agent

```bash
python main.py
```

---

## Specifications

*(To be filled in)*

---

## Roadmap

- **Optimize** the overall pipeline for better performance and lower latency  
- **Add CLI** support to allow more advanced control and flexibility  
