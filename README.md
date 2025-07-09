# Audio Transcription & Analysis Pipeline

This project performs end-to-end processing of voice memos or meeting recordings using Whisper, WhisperX, Hugging Face Transformers, spaCy, TextBlob, and OpenAI.

## 🔧 Features
- Automatic audio transcription using Whisper
- Speaker diarization with WhisperX
- Executive summary and meeting notes via BART summarization
- Rule-based and LLM-based action item extraction
- Keyword extraction with KeyBERT
- Named entity recognition and speaker sentiment analysis

## 🗂 Project Structure
```
Audio_Transcription_Automation/
├── main.py                        # Orchestrates the full pipeline
├── utils/
│   ├── __init__.py               # Package initializer
│   ├── transcription.py          # Whisper-based transcription
│   ├── summarization.py          # BART summarization & meeting notes
│   ├── speaker_tagging.py        # Adds synthetic speaker tags
│   ├── keywords.py               # Extracts keywords with KeyBERT
│   ├── action_items.py           # Detects actions (rule + LLM)
│   ├── analysis.py               # NER & sentiment analysis
├── output/                       # Saved results
│   ├── *.txt                     # Generated reports
├── .env                          # API keys (OPENAI_API_KEY, HF_TOKEN)
```

## 📦 Requirements
- Python 3.10+
- `torch`, `whisper`, `whisperx`
- `transformers`, `spacy`, `keybert`, `textblob`, `nltk`, `openai`, `ffmpeg`

## 📋 Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```

## 🧪 Usage
```bash
python main.py
```

## 📁 .env Example
```
OPENAI_API_KEY=sk-xxxx
HF_TOKEN=hf_xxxx
```

## 🧠 Notes
- `__init__.py`: marks the `utils/` folder as a Python package so you can import functions cleanly
- All outputs are saved in the `/output` directory

## 📬 Contact
Andy Collier · C Tech Solutions LLC
