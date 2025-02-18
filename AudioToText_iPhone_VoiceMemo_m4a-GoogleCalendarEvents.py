import os
import shutil
import whisper
from transformers import pipeline
from datetime import date
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import sys
import argparse  # For parsing command-line arguments

sys.stdout.reconfigure(encoding='utf-8')

# Google Calendar API Setup
SCOPES = ['https://www.googleapis.com/auth/calendar']
creds = None
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)
calendar_service = build('calendar', 'v3', credentials=creds)

# Prompt for input and output folders
#input_folder = input("Enter the path of the folder containing raw audio files: ").strip()
#output_folder = input("Enter the path of the output folder for processed files: ").strip()


# Command-line argument parser
parser = argparse.ArgumentParser(description="Transcribe audio files and create tasks.")
parser.add_argument('--input_folder', required=True, help='Path to the folder containing raw audio files')
parser.add_argument('--output_folder', required=True, help='Path to the output folder for processed files')

args = parser.parse_args()
input_folder = args.input_folder
output_folder = args.output_folder


# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

# Function to transcribe and combine audio files
def transcribe_and_combine(input_dir, output_dir):
    combined_text = ""
    audio_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".m4a")])
    
    for file in audio_files:
        result = whisper_model.transcribe(file)
        combined_text += result["text"] + "\n\n"
        print(f"Transcription completed for: {file}")
        # Move processed file to the output directory
        shutil.move(file, os.path.join(output_dir, os.path.basename(file)))
    return combined_text.strip()

# Function to create tasks based on the text
def create_google_tasks(text):
    # Summarize text into manageable tasks using Hugging Face summarization
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = summarizer([text[i:i+1024] for i in range(0, len(text), 1024)], max_length=150, min_length=60, do_sample=False)
    summarized_texts = [s['summary_text'] for s in summaries]

    # Create tasks and add them to Google Calendar
    for idx, summary in enumerate(summarized_texts):
        task_title = f"Task {idx + 1}: {summary[:50]}..."
        create_calendar_event(task_title, summary)

# Function to create a Google Calendar event
def create_calendar_event(title, description, target_date=None):
    if not target_date:
        target_date = date.today().isoformat()
    event = {
        'summary': title,
        'description': description,
        'start': {'date': target_date},
        'end': {'date': target_date},
    }
    event_result = calendar_service.events().insert(calendarId='primary', body=event).execute()
    print(f"Event created: {event_result['htmlLink']}")

# Transcribe audio files and create tasks
combined_text = transcribe_and_combine(input_folder, output_folder)
print("All transcriptions combined:")
print(combined_text)

# Generate tasks and add to Google Calendar
create_google_tasks(combined_text)
