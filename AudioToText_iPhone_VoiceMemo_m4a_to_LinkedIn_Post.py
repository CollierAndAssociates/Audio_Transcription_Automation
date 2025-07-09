# python -m venv venv
# .\venv\Scripts\activate
# python -m pip install --upgrade pip
# pip install torch
# pip install transformers
# pip install datasets huggingface_hub
# pip install git+https://github.com/openai/whisper.git

import whisper
from transformers import pipeline
import sys

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

# Function to create a LinkedIn post based on transcribed text
def generate_linkedin_post(text):
    # Load Hugging Face summarization model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Dynamically adjust summarization length based on input size
    # input_length = len(text.split())
    # max_len = min(150, max(60, input_length // 2))  # Scale max_length based on input size

    # # Summarize in chunks if needed
    # if len(text) > 1024:  
    #     summaries = summarizer([text[i:i+1024] for i in range(0, len(text), 1024)], 
    #                            max_length=max_len, min_length=max(30, max_len // 2), do_sample=False)
    #     summarized_text = " ".join([s['summary_text'] for s in summaries])
    # else:
    #     summarized_text = summarizer(text, max_length=max_len, min_length=max(30, max_len // 2), do_sample=False)[0]['summary_text']

    # Summarize the combined text
    summarized_text = summarizer(text, max_length=250, min_length=100, do_sample=False)[0]['summary_text']

    # Dynamically create a LinkedIn post with the summarized content
    linkedin_post = f"""
### **LinkedIn Post: From Custom ERP Solutions to AI-Enhanced Forecasting**  
🚀 For the first 25 years of my consulting career, I led countless successful transitions from legacy systems to ERP environments, designing custom solutions to address business challenges across industries like chemicals, manufacturing, and distribution. A key focus was creating algorithms and systems to manage large-scale, dynamic inventory across 18 national distribution centers and 250,000 active products.

💡 **Then:**  
{summarized_text}

💡 **Now:**  
Today’s **AI and ML advancements** make it possible to **go even further** by dynamically incorporating real-time data from multiple external sources, such as:  
- **Consumer sentiment analytics**  
- **Weather forecasts**  
- **Geopolitical risks**  
- **Competitor activities**  

🔍 **Modern AI technologies offer a powerful framework for automating demand forecasting:**  
- **ML-based time-series models (e.g., LSTM, Prophet)** to handle complex seasonality and product transitions.  
- **NLP tools** for extracting external signals, like social media sentiment or economic indicators.  
- **Generative AI** to produce actionable insights from large-scale data across internal systems and external events.  
- **Automated S&OP adjustments** that continuously optimize stock levels across distribution centers.  

🤖 **From Historical Systems to AI-Driven Optimization:**  
By merging traditional ERP logic with today’s **generative AI** and **deep learning pipelines**, businesses can automate the entire sales and operations planning (S&OP) process, detect demand anomalies, and deliver real-time, granular forecasts at scale.

---

### **Addendum: From Voice Memo to AI-Generated LinkedIn Post in 30 Minutes**  
This post started as an **iPhone Voice Memo** during a walk. Here’s the AI-powered workflow that turned raw audio into this polished content:  

💻 **The Tech Stack:**  
- **Whisper by OpenAI**: Converted the voice memo to text using state-of-the-art speech-to-text models.  
- **Hugging Face Transformers**: Summarized and enhanced the text using advanced NLP models.  
- **Python Libraries**: Included **Torch**, **Whisper**, and **Transformers** for seamless processing.  

This workflow demonstrates how AI is revolutionizing productivity and innovation—letting us focus on the bigger picture.  

Are you using AI in similar ways to reimagine your workflows and strategies? Let’s discuss! 👇
"""
    return linkedin_post.strip()
   # return linkedin_post

# Generate the final LinkedIn post
final_post = generate_linkedin_post(combined_text)

# Save the post to a file using UTF-8 encoding
with open("linkedin_post.txt", "w", encoding="utf-8") as f:
    f.write(final_post)

# Print the post to the console
print("\nGenerated LinkedIn Post:\n")
print(final_post)
