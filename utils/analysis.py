import re
from textblob import TextBlob

def analyze_entities_and_sentiment(tagged_text, nlp):
    entities = {}
    sentiments = {}

    for line in tagged_text.splitlines():
        if not line.strip():
            continue
        match = re.match(r"\[([^\]]+)\] (.+)", line)
        if not match:
            continue
        speaker, sentence = match.groups()

        doc = nlp(sentence)
        for ent in doc.ents:
            entities.setdefault(ent.label_, set()).add(ent.text)

        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity
        sentiments.setdefault(speaker, []).append(polarity)

    speaker_sentiment = {
        speaker: sum(vals) / len(vals) for speaker, vals in sentiments.items()
    }

    return entities, speaker_sentiment
