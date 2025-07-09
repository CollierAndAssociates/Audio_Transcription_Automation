from datetime import datetime
import re
import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime
from transformers import pipeline

#def format_summary(summary_text):
#    return f"""
#==============================
#Executive Summary Report
#==============================
#
#Date: {datetime.now().strftime('%B %d, %Y')}
#Prepared By: AI Meeting Assistant
#
#--- Summary ---
#{summary_text.strip()}
#"""

def format_summary(raw_text: str) -> str:
    """
    Formats a raw LLM-generated executive summary into a structured,
    boardroom-ready report with today's date.
    """
    current_date = datetime.now().strftime("%B %d, %Y")  # Example: July 09, 2025

    summary = [
        "==============================",
        "Executive Summary Report",
        "==============================",
        "",
        f"🗓️ Date: {current_date}",
        "👤 Prepared By: AI Meeting Assistant",
        "",
        "---",
        "",
        "🔹 Objective",
        "Brief one-line meeting objective here.",
        "",
        "---",
        "",
        "📌 Key Discussion Points",
        "- Bullet point key insights...",
        "",
        "---",
        "",
        "✅ Next Steps",
        "- Actionable steps with owners, if available...",
        "",
        "---",
        "",
        "🧠 Risks & Concerns",
        "- Known blockers, challenges, or decision delays...",
    ]
    return "\n".join(summary)

def format_notes(detailed_notes):
    bullets = detailed_notes.strip().split('\n')
    grouped = '\n'.join([f"  {line}" for line in bullets if line])
    return f"""
==============================
Detailed Meeting Notes
==============================

{grouped}
"""

def format_meeting_notes(raw_text):
    # Basic cleanup
    sentences = sent_tokenize(raw_text)
    cleaned = [re.sub(r"\b(you know|I mean|so|like|kind of|sort of|uh|um)\b", "", s, flags=re.IGNORECASE) for s in sentences]
    
    # Chunk into ~4-line groups
    chunk_size = 4
    chunks = [cleaned[i:i+chunk_size] for i in range(0, len(cleaned), chunk_size)]

    output = ["==============================", "Detailed Meeting Notes", "==============================", ""]
    for i, chunk in enumerate(chunks):
        output.append(f"🟦 Topic {i+1}")
        for s in chunk:
            output.append(f"- {s.strip()}")
        output.append("")  # spacing
    
    return "\n".join(output)

def format_action_items(rule_based=None, llm_based=None):
    output = "==============================\nAction Items Summary\n==============================\n"
    if rule_based:
        output += "\n--- Rule-Based Detections ---\n"
        for i, item in enumerate(rule_based, 1):
            output += f"{i}. {item}\n"
    if llm_based:
        output += "\n--- LLM-Derived Items ---\n"
        output += llm_based.strip() + "\n"
    return output

def clean_action_sentences(sentences):
    cleaned = []
    for s in sentences:
        s = s.strip("- ").strip()
        if len(s) < 30:
            continue  # skip trivial lines
        if re.search(r"(make sure|need to|should|must|schedule|send|validate|review|follow up|assign|ensure)", s, re.IGNORECASE):
            # Normalize
            s = re.sub(r"\\b(I think|just|you know|so|okay|well|yeah|right|um)\\b", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\\s{2,}", " ", s).strip()
            # Add placeholder for ownership
            cleaned.append(f"- [Owner: TBD] {s}")
    return cleaned

#def format_keywords(keyword_list):
#    return f"""
#==============================
#Key Discussion Themes
#==============================
#
#{chr(8226)} " + f"\n{chr(8226)} ".join(keyword_list)
#"""

def format_keywords(keyword_list):
    bullets = "\n".join(f"• {kw}" for kw in keyword_list)
    return f"""\
==============================
Key Discussion Themes
==============================

{bullets}
"""

def format_entities_and_sentiment(entities, sentiment_dict):
    entity_block = "--- Named Entities ---\n"
    for label, items in entities.items():
        entity_block += f"{label}: {', '.join(sorted(items))}\n"

    sentiment_block = "\n--- Speaker Sentiment ---\n"
    for speaker, score in sentiment_dict.items():
        tone = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
        sentiment_block += f"{speaker}: {tone} ({score:.2f})\n"

    return f"""
==============================
Entity & Sentiment Overview
==============================

{entity_block}{sentiment_block}
"""

corrector = pipeline("text2text-generation", model="google/flan-t5-base")

def correct_transcript_line(line: str) -> str:
    prefix, text = line.split("] ", 1)
    corrected = corrector(f"Fix grammar and clarity: {text}", max_length=128)[0]['generated_text']
    return f"{prefix}] {corrected}"

def enhance_tagged_transcript(raw_text: str, speaker_map: dict = None) -> str:
    import re

    lines = raw_text.splitlines()
    enhanced_lines = []

    for line in lines:
        # Skip trivial filler lines
        if re.fullmatch(r"\[.*?\]\s*(yeah|yes|uh|um|right|okay|just in)[.!]?", line.strip(), re.I):
            continue

        # Apply speaker mapping if available
        if speaker_map:
            match = re.match(r"\[(Speaker \d+)\]", line)
            if match:
                original = match.group(1)
                name = speaker_map.get(original, original)
                line = line.replace(original, name)

        enhanced_lines.append(line)

    return "\n".join(enhanced_lines)
