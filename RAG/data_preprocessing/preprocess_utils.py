import re
from bs4 import BeautifulSoup
import spacy


def count_sentences(text):
    # Load the Spacy model
    nlp = spacy.load("en_core_web_sm")  
    doc = nlp(text)
    
    return len(list(doc.sents))

def sanitize_filename(filename):
    # Replace invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def preprocess_text(text):
    try:
        text = text.encode('utf-8').decode('unicode_escape', errors='replace')
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
        text = text.encode('utf-8', errors='replace').decode('unicode_escape', errors='replace')
    return text

def clean_text(text):
    # Define the unwanted characters
    unwanted_prefix = "\u00c3\u00af\u00c2\u00bb\u00c2\u00bf"
    if text.startswith(unwanted_prefix):
        return text[len(unwanted_prefix):]
    return text