import re
import unicodedata

def normalize_unicode(text):
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFC', text)

def remove_noise(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)                 
    text = re.sub(r'http\S+|www\S+', '', text)         
    text = re.sub(r'[^a-zA-Z\s]', '', text)            
    text = re.sub(r'\s+', ' ', text).strip()          
    return text

def apply_case_folding(text):
    if not isinstance(text, str):
        return ""
    return text.lower()

def core_clean_text(text):
    text = normalize_unicode(text)
    text = remove_noise(text)
    text = apply_case_folding(text)
    return text