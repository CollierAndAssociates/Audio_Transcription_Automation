import re
import nltk
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

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