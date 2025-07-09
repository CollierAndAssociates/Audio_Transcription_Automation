def generate_meeting_notes(text, summarizer, max_tokens=900):
    def chunk_text(text, max_tokens):
        sentences = text.split('. ')
        chunks, chunk = [], ""
        for sentence in sentences:
            if len(chunk) + len(sentence) <= max_tokens:
                chunk += sentence + ". "
            else:
                chunks.append(chunk.strip())
                chunk = sentence + ". "
        if chunk:
            chunks.append(chunk.strip())
        return chunks

    chunks = chunk_text(text, max_tokens)
    summaries = [summarizer(chunk, max_length=250, min_length=100, do_sample=False)[0]['summary_text'] for chunk in chunks]
    executive_summary = "\n".join(summaries)

    bullets = [f"- {line.strip()}" for line in text.splitlines() if line.strip()]
    detailed_notes = "\n".join(bullets)

    return executive_summary.strip(), detailed_notes.strip()