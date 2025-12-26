## 🚀 Setup & Installation
**Prerequisites:** Python 3.10+, Ollama installed and running locally, optional virtual environment, and `ffmpeg` for audio.

1) **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```
2) **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3) **Download model**
   - Grab `UK_Law_Real_Final.gguf` from [Google Drive](https://drive.google.com/file/d/1DTs8bWrYydRTBkSZhGS9Vm9Xz9-Koqu5/view?usp=sharing) and place it in the project root.
4) **Build and load models (Ollama)**
   ```bash
   ollama create uk-lawyer -f Modelfile
   ollama pull nomic-embed-text
   ```
5) **Launch the application**
   ```bash
   streamlit run app/lawyer_gui.py
   ```

## ✨ Capabilities
- Specialized UK Contract Law knowledge using the IRAC framework.
- Reads statutes (e.g., Consumer Rights Act 2015) before replying.
- Automatic routing for Goods, Services, and Digital Content scenarios.
- Privacy-first: all processing stays local via Ollama.

## 🎙️ Voice Interaction
- Mic widget for dictation; SpeechRecognition handles local transcription.
- Each response includes a **Read Aloud** button powered by gTTS.
- Install `ffmpeg` (e.g., `brew install ffmpeg`) for audio conversions.

## 📂 Legal Documents
Organize PDFs under `legal_docs/` by domain so they can be indexed:
```
legal_docs/
├── Misrepresentation/
│   ├── Misrepresentation.pdf
│   └── Misrepresentation-Act-1967.pdf
├── Contractual Terms/
│   ├── Breach of Contract.pdf
│   └── Contractual Terms.pdf
├── Offer and Acceptance/
│   ├── ukpga_20150015_en.pdf
│   └── Offer, Acceptance.pdf
├── Promissory Estoppel/
│   └── Intention, Consideration and Promissory Estoppel.pdf
└── Mistake - Mutual Mistake/
    └── Mistake.pdf
```

## 🛠️ Troubleshooting
- “No PDFs found”: check for trailing spaces in folder names, restart Streamlit (`Ctrl+C`, then `streamlit run app/lawyer_gui.py`), click **Rebuild Graph**, then clear caches and retry.
- Audio issues: ensure `ffmpeg` is installed and mic permissions are granted.

## ⚖️ Disclaimer
This is an AI research tool, not a substitute for a qualified solicitor. Always verify legal citations and consult professionals for legal decisions.