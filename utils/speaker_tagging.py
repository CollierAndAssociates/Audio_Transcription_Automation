import nltk

def add_speaker_tags(text):
    lines = nltk.sent_tokenize(text)
    tagged = []
    speaker = 1
    for i, line in enumerate(lines):
        if i % 5 == 0:
            speaker += 1
        tagged.append(f"[Speaker {speaker}] {line}")
    return "\n".join(tagged)
