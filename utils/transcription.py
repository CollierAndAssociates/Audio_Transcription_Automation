import whisper
import logging

def transcribe_audio_files(audio_files, model_size="base", device="cpu"):
    model = whisper.load_model(model_size, device=device)
    combined_text = ""
    for file in audio_files:
        try:
            result = model.transcribe(file)
            combined_text += result["text"] + "\n\n"
            logging.info(f"Transcription completed for: {file}")
        except Exception as e:
            logging.error(f"Error transcribing {file}: {e}")
    return combined_text.strip()