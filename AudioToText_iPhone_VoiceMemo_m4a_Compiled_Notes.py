# python -m venv venv
# .\venv\Scripts\activate
# python -m pip install --upgrade pip
# pip install torch
# pip install transformers
# pip install datasets huggingface_hub
# pip install git+https://github.com/openai/whisper.git
# pip install keybert sentence-transformers nltk
#
# Install CUDA
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install git+https://github.com/openai/whisper.git
#
# pip install -U whisperx
# pip install openai
#
# pip install spacy
# python -m spacy download en_core_web_sm
# pip install textblob
# python -m textblob.download_corpora


import whisper
from transformers import pipeline
import sys
from transformers import pipeline
from keybert import KeyBERT
import nltk
import re
import whisperx


import openai
import os

import spacy
from textblob import TextBlob

openai.api_key = os.getenv("OPENAI_API_KEY")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
sys.stdout.reconfigure(encoding='utf-8')

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

# List of audio file paths to process
audio_files = [
    r"C:\Users\andy\OneDrive - Collier & Associates\CA-Code\Repositories_Files\Audio_Transcription_Automation_Files\Audio\100-Unprocessed\New Recording 33.m4a"
]

# Function to transcribe all audio files and combine the text
def transcribe_and_combine(files):
    combined_text = ""
    for file in files:
        result = whisper_model.transcribe(file)
        combined_text += result["text"] + "\n\n"
        print(f"Transcription completed for: {file}")
    return combined_text.strip()

# Transcribe and combine the audio files
combined_text = transcribe_and_combine(audio_files)

print("All transcriptions combined:")
print(combined_text)

# Function to generate meeting summary and notes
def generate_meeting_notes(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Split long input into manageable chunks
    def chunk_text(text, max_tokens=900):
        sentences = text.split('. ')
        chunks, chunk = [], ""
        for sentence in sentences:
            if len(chunk) + len(sentence) <= max_tokens:
                chunk += sentence + ". "
            else:
                chunks.append(chunk.strip())
                chunk = sentence + ". "
        if chunk:
            chunks.append(chunk.strip())
        return chunks

    # Summarize key points
    chunks = chunk_text(text)
    summaries = [summarizer(chunk, max_length=250, min_length=100, do_sample=False)[0]['summary_text'] for chunk in chunks]

    # Create executive summary
    executive_summary = "\n".join(summaries)

    # Convert full transcript into more readable notes
    bullets = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 0:
            # Turn long statements into bullet-format ideas
            bullets.append(f"- {line}")

    detailed_notes = "\n".join(bullets)

    return executive_summary.strip(), detailed_notes.strip()

# Generate executive summary and detailed notes
executive_summary, detailed_notes = generate_meeting_notes(combined_text)

# Save to separate files
with open("meeting_summary.txt", "w", encoding="utf-8") as f:
    f.write("=== Executive Summary ===\n\n")
    f.write(executive_summary)

with open("meeting_notes.txt", "w", encoding="utf-8") as f:
    f.write("=== Detailed Meeting Notes ===\n\n")
    f.write(detailed_notes)

# Output to console
print("\n=== Executive Summary ===\n")
print(executive_summary)
print("\n=== Detailed Meeting Notes ===\n")
print(detailed_notes)

nltk.download("punkt")

# === Speaker Attribution ===
def add_speaker_tags(text):
    lines = nltk.sent_tokenize(text)
    tagged = []
    speaker = 1
    for i, line in enumerate(lines):
        if i % 5 == 0:  # Tag every 5 sentences as a new speaker
            speaker += 1
        tagged.append(f"[Speaker {speaker}] {line}")
    return "\n".join(tagged)

tagged_transcript = add_speaker_tags(combined_text)
with open("tagged_transcript.txt", "w", encoding="utf-8") as f:
    f.write("=== Transcript with Speaker Tags ===\n\n")
    f.write(tagged_transcript)

# === Keyword Extraction ===
def extract_keywords(text, top_n=20):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words='english')
    return [kw[0] for kw in keywords]

keywords = extract_keywords(combined_text)
with open("keywords.txt", "w", encoding="utf-8") as f:
    f.write("=== Top Keywords ===\n\n")
    f.write("\n".join(keywords))

# === Action Item Detection ===
def extract_action_items(text):
    sentences = nltk.sent_tokenize(text)
    action_items = []
    action_starters = [
        r"\b(we|you|I)\s+(need to|will|should|must|can)\b",
        r"\blet's\b",
        r"\bplease\b",
        r"\bassign\b",
        r"\bset up\b",
        r"\bschedule\b",
        r"\bmake sure\b",
        r"\bensure\b"
    ]
    pattern = re.compile("|".join(action_starters), flags=re.IGNORECASE)
    for sentence in sentences:
        if pattern.search(sentence):
            action_items.append("- " + sentence.strip())
    return action_items

actions = extract_action_items(combined_text)
with open("action_items.txt", "w", encoding="utf-8") as f:
    f.write("=== Action Items ===\n\n")
    if actions:
        f.write("\n".join(actions))
    else:
        f.write("No action items detected.")
        
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("base", device)

for audio_path in audio_files:
    result = model.transcribe(audio_path)
    
    # Run diarization
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=True)  # HF token required
    diarize_segments = diarize_model(audio_path)
    
    # Align speakers with text
    result = whisperx.assign_word_speakers(diarize_segments, result["segments"])
    
    # Save speaker-tagged transcript
    with open("whisperx_speaker_transcript.txt", "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            speaker = segment.get("speaker", "Unknown")
            f.write(f"[{speaker}] {segment['text'].strip()}\n")
            
def extract_actions_llm(text):
    prompt = f"""You are an expert meeting assistant. Extract all action items from this conversation transcript.
        Return them as a bullet list. Be concise and only include real commitments or follow-ups.

Transcript:
{text}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

llm_actions = extract_actions_llm(combined_text)
with open("action_items_llm.txt", "w", encoding="utf-8") as f:
    f.write("=== LLM-Derived Action Items ===\n\n")
    f.write(llm_actions)

nlp = spacy.load("en_core_web_sm")

def analyze_entities_and_sentiment(tagged_text):
    entities = {}
    sentiments = {}
    
    for line in tagged_text.splitlines():
        if not line.strip(): continue
        match = re.match(r"\[([^\]]+)\] (.+)", line)
        if not match: continue
        speaker, sentence = match.groups()
        
        # NER
        doc = nlp(sentence)
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = set()
            entities[ent.label_].add(ent.text)
        
        # Sentiment
        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity
        sentiments.setdefault(speaker, []).append(polarity)

    # Avg sentiment per speaker
    speaker_sentiment = {
        speaker: sum(vals)/len(vals) for speaker, vals in sentiments.items()
    }

    return entities, speaker_sentiment

entities, sentiment = analyze_entities_and_sentiment(tagged_transcript)

with open("entities_sentiment.txt", "w", encoding="utf-8") as f:
    f.write("=== Named Entities ===\n\n")
    for label, ents in entities.items():
        f.write(f"{label}: {', '.join(sorted(ents))}\n")
    
    f.write("\n=== Speaker Sentiment ===\n\n")
    for speaker, score in sentiment.items():
        f.write(f"{speaker}: {'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'} ({score:.2f})\n")
          