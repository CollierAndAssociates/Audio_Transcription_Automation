from datetime import datetime

def format_summary(summary_text):
    return f"""
==============================
Executive Summary Report
==============================

Date: {datetime.now().strftime('%B %d, %Y')}
Prepared By: AI Meeting Assistant

--- Summary ---
{summary_text.strip()}
"""

def format_notes(detailed_notes):
    bullets = detailed_notes.strip().split('\n')
    grouped = '\n'.join([f"  {line}" for line in bullets if line])
    return f"""
==============================
Detailed Meeting Notes
==============================

{grouped}
"""

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

def format_keywords(keyword_list):
    return f"""
==============================
Key Discussion Themes
==============================

{chr(8226)} " + f"\n{chr(8226)} ".join(keyword_list)
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
