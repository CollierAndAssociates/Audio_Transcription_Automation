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
import argparse
import nltk  # For sentence tokenization (requires `pip install nltk`)
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

sys.stdout.reconfigure(encoding='utf-8')

# Google Tasks API Setup
SCOPES = ['https://www.googleapis.com/auth/tasks']
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
tasks_service = build('tasks', 'v1', credentials=creds)

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

# Function to split the transcribed text into specific, actionable tasks
def split_into_tasks(text):
    # Tokenize into sentences using NLTK
    sentences = sent_tokenize(text)

    # Use summarization to extract key tasks
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summarized_sentences = summarizer(
        [text[i:i+1024] for i in range(0, len(text), 1024)], 
        max_length=150, min_length=60, do_sample=False
    )
    summarized_text = " ".join([s['summary_text'] for s in summarized_sentences])

    # Split summarized content into actionable tasks
    task_candidates = sent_tokenize(summarized_text)
    return task_candidates

# Function to create tasks in the specified Google Task List
def create_google_task(task_list_id, title, notes, due_date=None):
    task_body = {
        'title': title,
        'notes': notes,
        'due': f"{due_date}T00:00:00.000Z" if due_date else None
    }
    result = tasks_service.tasks().insert(tasklist=task_list_id, body=task_body).execute()
    print(f"Task created: {result['title']}")

# Get the task list ID for "AC" or create if not found
def get_or_create_task_list(list_name='AC'):
    # Retrieve existing task lists
    task_lists = tasks_service.tasklists().list().execute().get('items', [])
    for task_list in task_lists:
        if task_list['title'] == list_name:
            return task_list['id']
    
    # If not found, create a new task list
    new_list = tasks_service.tasklists().insert(body={'title': list_name}).execute()
    print(f"Created new task list: {new_list['title']}")
    return new_list['id']

# Main function to generate tasks from combined text
def generate_google_tasks_from_text(combined_text):
    task_list_id = get_or_create_task_list()  # Default task list is "AC"
    tasks = split_into_tasks(combined_text)

    if not tasks:
        print("No tasks identified.")
        return

    # Assign tasks to the task list
    for idx, task in enumerate(tasks):
        title = f"Task {idx + 1}: {task[:50]}..."  # Task title with up to 50 characters
        create_google_task(task_list_id, title, task, date.today().isoformat())

# Transcribe audio files and create tasks
combined_text = transcribe_and_combine(input_folder, output_folder)
print("All transcriptions combined:")
print(combined_text)

# Generate tasks and add them to Google Tasks
generate_google_tasks_from_text(combined_text)
