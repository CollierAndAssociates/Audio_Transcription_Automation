from keybert import KeyBERT

def extract_keywords(text, top_n=20):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words='english')
    return [kw[0] for kw in keywords]
