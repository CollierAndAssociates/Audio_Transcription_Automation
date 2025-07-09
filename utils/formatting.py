# utils/formatting.py

from datetime import datetime
from typing import List, Dict
import re

def format_summary(summary: str) -> str:
    return (
        f"==============================\n"
        f"Executive Summary Report\n"
        f"==============================\n\n"
        f"Date: {datetime.now().strftime('%B %d, %Y')}\n"
        f"Prepared By: AI Meeting Assistant\n\n"
        f"--- Summary ---\n{summary.strip()}\n"
    )

def format_meeting_notes(notes: str) -> str:
    return (
        f"==============================\n"
        f"Detailed Meeting Notes\n"
        f"==============================\n\n"
        f"{notes.strip()}\n"
    )

def format_keywords(keyword_list: List[str]) -> str:
    return (
        f"==============================\n"
        f"Key Discussion Themes\n"
        f"==============================\n\n"
        + "\n• ".join(["• " + kw for kw in keyword_list])
    )

def format_action_items(rule_based: List[str] = None, llm_based: str = None) -> str:
    output = ["==============================", "Action Items Summary", "==============================\n"]

    if rule_based:
        output.append("--- Rule-Based Detections ---")
        for i, item in enumerate(rule_based, 1):
            output.append(f"{i}. {item}")
        output.append("")

    if llm_based:
        output.append("--- LLM-Derived Actions ---\n")
        output.append(llm_based.strip())

    return "\n".join(output)

def format_entities_and_sentiment(entities: Dict[str, set], sentiment: Dict[str, float]) -> str:
    result = ["==============================", "Entity & Sentiment Analysis", "==============================\n"]

    result.append("=== Named Entities ===\n")
    for label, ents in entities.items():
        result.append(f"{label}: {', '.join(sorted(ents))}\n")

    result.append("\n=== Speaker Sentiment ===\n")
    for speaker, score in sentiment.items():
        sentiment_label = (
            "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
        )
        result.append(f"{speaker}: {sentiment_label} ({score:.2f})")

    return "\n".join(result)

def clean_action_sentences(actions: List[str]) -> List[str]:
    cleaned = []
    for line in actions:
        line = re.sub(r"^[-–\s]*", "", line)  # remove leading bullets/dashes
        line = re.sub(r"\s+", " ", line).strip()  # collapse extra whitespace
        if line and len(line.split()) >= 3:  # filter short or junk entries
            cleaned.append(line)
    return cleaned

def enhanced_tagged_transcript(raw_text: str, speaker_map: Dict[str, str] = None) -> str:
    """Optionally replaces speaker tags and removes junk lines."""
    output_lines = []
    lines = raw_text.splitlines()
    for line in lines:
        if line.startswith("[Speaker"):
            match = re.match(r"\[Speaker (\d+)\](.*)", line)
            if match:
                speaker_id = f"Speaker {match.group(1)}"
                text = match.group(2).strip()
                if len(text.split()) < 3:
                    continue  # Skip short filler
                if speaker_map and speaker_id in speaker_map:
                    line = f"[{speaker_map[speaker_id]}] {text}"
                else:
                    line = f"[{speaker_id}] {text}"
        output_lines.append(line)
    return "\n".join(output_lines)