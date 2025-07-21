# 🤖 Jarvis AI Assistant

Jarvis is a powerful, modular voice-activated AI assistant built in Python. It supports natural language interaction, voice recognition, intent detection, and task execution through a combination of fuzzy logic, spaCy NLP, and a neural model. It includes hot-reloading, multi-threaded processing, and real-time command execution.

---

## 📁 Project Structure

```
JarvisAI/
├── jarvis.py       # Main assistant logic
├── task.py         # Intent-to-action handler
├── intents.json    # Custom intents
├── requirements.txt
└── README.md
```

---

## 🚀 Features

- ✅ Voice and text interaction
- ✅ Confidence-weighted intent detection (fuzzy + spaCy + neural net)
- ✅ Real-time TTS using `edge-tts`
- ✅ Fast transcription via `faster-whisper`
- ✅ Hot-reload support for `intents.json`
- ✅ Context tracking for follow-ups
- ✅ Task routing via `task.py`
- ✅ Parallel intent matching for speed
- ✅ Alarm, screenshot, calculator, and OS control tasks
- ✅ AI-powered symptom-based disease diagnosis

---

## 🧠 Intent Detection

Each input is evaluated using:
- `fuzz.ratio` (RapidFuzz)
- spaCy similarity
- Neural network (custom `predict_nn()`)

The result is voted using weighted confidence to select the best match.

---

## 🧪 Sample Intents (`intents.json`)
```json
{
  "intents": [
    {
      "tag": "open_google",
      "patterns": ["open google", "launch google", "search engine"],
      "responses": ["Opening Google..."]
    },
    {
      "tag": "diagnose_symptoms",
      "patterns": ["I feel sick", "I have a cold", "symptom check"],
      "responses": ["Analyzing your symptoms..."]
    }
  ]
}
```

---

## 🛠️ Setup

### 1. Clone this repo
```bash
git clone https://github.com/ChinmayBansal010/jarvis.git
cd jarvis
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install external models
```bash
python -m spacy download en_core_web_sm
```

---

## ▶️ Run Jarvis
```bash
python jarvis.py
```

---

## 🧩 Customize
- Modify `intents.json` to add new intents
- Define new behaviors in `task.py`
- Add fallback handling for unmatched queries

---

## 📋 Requirements
See `requirements.txt`, includes:
- `faster-whisper`, `edge-tts`, `pvporcupine`, `webrtcvad`, `spacy`, `rapidfuzz`, `watchdog`, etc.

---

## 🛡️ License
MIT License. You are free to use and modify this project with attribution.

---

## 💬 Credits
Built by Chinmay Bansal

