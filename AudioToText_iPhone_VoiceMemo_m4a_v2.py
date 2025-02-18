# Install necessary libraries
# pip install openai-whisper torch transformers

import whisper
from transformers import pipeline
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

# List of audio file paths to process
audio_files = [
    r"C:\Users\andy\OneDrive - Collier & Associates\CA-Code\Excel\VendorRFI&DemoEvals\13-iphone-VoiceMemo.m4a",
    r"C:\Users\andy\OneDrive - Collier & Associates\CA-Code\Excel\VendorRFI&DemoEvals\14-iphone-VoiceMemo.m4a",
    r"C:\Users\andy\OneDrive - Collier & Associates\CA-Code\Excel\VendorRFI&DemoEvals\15-iphone-VoiceMemo.m4a"
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

# Function to create a LinkedIn post based on transcribed text
def generate_linkedin_post(text):
    # Load Hugging Face summarization model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Dynamically adjust summarization length based on input size
    input_length = len(text.split())
    max_len = min(150, max(60, input_length // 2))  # Scale max_length based on input size

    # Summarize in chunks if needed
    if len(text) > 1024:  
        summaries = summarizer([text[i:i+1024] for i in range(0, len(text), 1024)], 
                               max_length=max_len, min_length=max(30, max_len // 2), do_sample=False)
        summarized_text = " ".join([s['summary_text'] for s in summaries])
    else:
        summarized_text = summarizer(text, max_length=max_len, min_length=max(30, max_len // 2), do_sample=False)[0]['summary_text']

    # Dynamically create a LinkedIn post with the summarized content
    linkedin_post = f"""
### **LinkedIn Post: Harnessing AI for Creative Problem-Solving**  
🚀 Over the years, I’ve learned how creative problem-solving can take software development and business solutions to the next level. Here’s a key takeaway from my recent reflections:  

💡 **Summary of My Insights:**  
{summarized_text}  

---  

### **Addendum: From Voice Memo to AI-Generated LinkedIn Post in 30 Minutes**  
This post started as an **iPhone Voice Memo** during a walk. Here’s the AI-powered workflow that turned raw audio into this polished content:  

💻 **The Tech Stack:**  
- **Whisper by OpenAI**: Converted the voice memo to text using state-of-the-art speech-to-text models.  
- **Hugging Face Transformers**: Summarized and enhanced the text using advanced NLP models.  
- **Python Libraries**: Included **Torch**, **Whisper**, and **Transformers** for seamless processing.  

This workflow demonstrates how AI is boosting productivity and creativity in ways I couldn’t have imagined just a few years ago.  

Are you using AI in similar ways to amplify your work? I’d love to hear your stories! 👇  
"""
    return linkedin_post

# Generate the final LinkedIn post
final_post = generate_linkedin_post(combined_text)

# Save the post to a file using UTF-8 encoding
with open("linkedin_post.txt", "w", encoding="utf-8") as f:
    f.write(final_post)

# Print the post to the console
print("\nGenerated LinkedIn Post:\n")
print(final_post)
