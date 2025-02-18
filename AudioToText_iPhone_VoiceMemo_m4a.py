# Install libraries
# pip install openai-whisper torch transformers

import whisper
import transformers
from transformers import pipeline
print(transformers.__version__)
# Load Whisper model (base version for demonstration, use larger models if needed)
whisper_model = whisper.load_model("base")

# Transcribe the audio file (replace with your file path)
result = whisper_model.transcribe(r"C:\Users\andy\OneDrive - Collier & Associates\CA-Code\Excel\VendorRFI&DemoEvals\iphone-VoiceMemo.m4a")
transcribed_text = result["text"]

print("Transcription complete!")
print(f"Full text: {transcribed_text}\n")

# Summarize the transcribed text using Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Split long transcriptions into manageable chunks if necessary
summary = summarizer(transcribed_text, max_length=130, min_length=50, do_sample=False)

print("Summary:")
print(summary[0]['summary_text'])
