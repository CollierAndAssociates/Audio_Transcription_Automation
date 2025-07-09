import os
import sys
import logging
import re
import nltk
from datetime import datetime

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
from utils.formatting import (
    format_summary,
    format_meeting_notes,
    format_action_items,
    format_keywords,
    format_entities_and_sentiment,
    clean_action_sentences,
    enhanced_tagged_transcript
)

# Setup
nltk.download("punkt")
logging.basicConfig(level=logging.INFO)
openai.api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Device set to use: {device}")

# Inputs
audio_files = [
    r"C:\\Users\\andy\\OneDrive - Collier & Associates\\CA-Code\\Repositories_Files\\Audio_Transcription_Automation_Files\\Audio\\100-Unprocessed\\New Recording 33.m4a"
]

# Step 1: Transcription
combined_text = transcribe_audio_files(audio_files, model_size="base", device=device)

# Step 2: Summarization and Notes
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
executive_summary, detailed_notes = generate_meeting_notes(combined_text, summarizer)

# Save formatted summary and notes
os.makedirs("output", exist_ok=True)
try:
    with open("output/meeting_summary.txt", "w", encoding="utf-8") as f:
        f.write(format_summary(executive_summary))
        logging.info("Wrote meeting_summary.txt")

    with open("output/meeting_notes.txt", "w", encoding="utf-8") as f:
        f.write(format_meeting_notes(detailed_notes))
        logging.info("Wrote meeting_notes.txt")
except Exception as e:
    logging.error(f"Failed writing summary or notes: {e}")

# Step 3: Speaker Tagging
try:
    tagged_transcript = add_speaker_tags(combined_text)
    tagged_transcript = enhanced_tagged_transcript(tagged_transcript)
    with open("output/tagged_transcript.txt", "w", encoding="utf-8") as f:
        f.write("=== Transcript with Speaker Tags ===\n\n" + tagged_transcript)
        logging.info("Wrote tagged_transcript.txt")
except Exception as e:
    logging.error(f"Speaker tagging failed: {e}")

# Step 4: Keyword Extraction
try:
    keywords = extract_keywords(combined_text, top_n=20)
    with open("output/keywords.txt", "w", encoding="utf-8") as f:
        f.write(format_keywords(keywords))
        logging.info("Wrote keywords.txt")
except Exception as e:
    logging.error(f"Keyword extraction failed: {e}")

# Step 5: Action Item Detection
try:
    rule_based_actions = extract_action_items(combined_text)
    cleaned_actions = clean_action_sentences(rule_based_actions)
    with open("output/action_items.txt", "w", encoding="utf-8") as f:
        f.write(format_action_items(rule_based=cleaned_actions))
        logging.info("Wrote action_items.txt")
except Exception as e:
    logging.error(f"Action item extraction failed: {e}")

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
        logging.info("Wrote whisperx_speaker_transcript.txt")
except Exception as e:
    logging.error(f"WhisperX diarization failed: {e}")

# Step 7: LLM-Based Action Extraction
try:
    llm_actions = extract_actions_llm(combined_text)
    with open("output/action_items_llm.txt", "w", encoding="utf-8") as f:
        f.write(format_action_items(llm_based=llm_actions))
        logging.info("Wrote action_items_llm.txt")
except Exception as e:
    logging.error(f"LLM-based action item extraction failed: {e}")

# Step 8: Entity and Sentiment Analysis
try:
    nlp = spacy.load("en_core_web_sm")
    entities, sentiment = analyze_entities_and_sentiment(tagged_transcript, nlp)
    with open("output/entities_sentiment.txt", "w", encoding="utf-8") as f:
        f.write(format_entities_and_sentiment(entities, sentiment))
        logging.info("Wrote entities_sentiment.txt")
except Exception as e:
    logging.error(f"Entity/sentiment analysis failed: {e}")
