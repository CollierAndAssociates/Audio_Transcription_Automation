import os
import sys
import logging
import re
import nltk

from transformers import pipeline
from keybert import KeyBERT
from textblob import TextBlob
import spacy
import openai
import torch

import whisper
import whisperx

from utils.transcription import transcribe_audio_files
from utils.summarization import generate_meeting_notes
from utils.speaker_tagging import add_speaker_tags
from utils.keywords import extract_keywords
from utils.action_items import extract_action_items, extract_actions_llm
from utils.analysis import analyze_entities_and_sentiment

# Setup
nltk.download("punkt")
logging.basicConfig(level=logging.INFO)
openai.api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Inputs
audio_files = [
    r"C:\\Users\\andy\\OneDrive - Collier & Associates\\CA-Code\\Repositories_Files\\Audio_Transcription_Automation_Files\\Audio\\100-Unprocessed\\New Recording 33.m4a"
]

# Step 1: Transcription
combined_text = transcribe_audio_files(audio_files, model_size="base", device=device)

# Step 2: Summarization and Notes
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
executive_summary, detailed_notes = generate_meeting_notes(combined_text, summarizer)

# Save summary and notes
os.makedirs("output", exist_ok=True)
with open("output/meeting_summary.txt", "w", encoding="utf-8") as f:
    f.write("=== Executive Summary ===\n\n" + executive_summary)
with open("output/meeting_notes.txt", "w", encoding="utf-8") as f:
    f.write("=== Detailed Meeting Notes ===\n\n" + detailed_notes)

# Step 3: Speaker Tagging
tagged_transcript = add_speaker_tags(combined_text)
with open("output/tagged_transcript.txt", "w", encoding="utf-8") as f:
    f.write("=== Transcript with Speaker Tags ===\n\n" + tagged_transcript)

# Step 4: Keyword Extraction
keywords = extract_keywords(combined_text, top_n=20)
with open("output/keywords.txt", "w", encoding="utf-8") as f:
    f.write("=== Top Keywords ===\n\n" + "\n".join(keywords))

# Step 5: Action Item Detection
actions = extract_action_items(combined_text)
with open("output/action_items.txt", "w", encoding="utf-8") as f:
    f.write("=== Action Items ===\n\n" + ("\n".join(actions) if actions else "No action items detected."))

# Step 6: WhisperX Diarization
try:
    model = whisperx.load_model("base", device)
    for audio_path in audio_files:
        result = model.transcribe(audio_path)
        if not result.get("segments"):
            logging.warning(f"No segments returned for {audio_path}")
            continue

        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token)
        diarize_segments = diarize_model(audio_path)
        result = whisperx.assign_word_speakers(diarize_segments, result["segments"])

        with open("output/whisperx_speaker_transcript.txt", "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                speaker = segment.get("speaker", "Unknown")
                f.write(f"[{speaker}] {segment['text'].strip()}\n")
except Exception as e:
    logging.error(f"WhisperX diarization failed: {e}")

# Step 7: LLM-Based Action Extraction
llm_actions = extract_actions_llm(combined_text)
with open("output/action_items_llm.txt", "w", encoding="utf-8") as f:
    f.write("=== LLM-Derived Action Items ===\n\n" + llm_actions)

# Step 8: Entity and Sentiment Analysis
nlp = spacy.load("en_core_web_sm")
entities, sentiment = analyze_entities_and_sentiment(tagged_transcript, nlp)
with open("output/entities_sentiment.txt", "w", encoding="utf-8") as f:
    f.write("=== Named Entities ===\n\n")
    for label, ents in entities.items():
        f.write(f"{label}: {', '.join(sorted(ents))}\n")
    f.write("\n=== Speaker Sentiment ===\n\n")
    for speaker, score in sentiment.items():
        f.write(f"{speaker}: {'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'} ({score:.2f})\n")