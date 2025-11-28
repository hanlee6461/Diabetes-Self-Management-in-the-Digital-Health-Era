import os
import re
import pathlib
import spacy
import enchant

# spaCy
nlp = spacy.load("en_core_web_sm")
dictionary = enchant.Dict("en_US")

# 1. Line break elimination
def fix_line_breaks(text):
    text = re.sub(r"-\s*\n\s*", "", text)                        
    text = re.sub(r"(?<![\.\?\!])\n+", " ", text)              
    return text

# 2. Meaningful hyphen-word protection
def protect_meaningful_hyphen_words(text):
    meaningful_prefixes = ["self", "mobile", "health", "patient", "e", "tele", "non"]
    pattern = r"\b(" + "|".join(meaningful_prefixes) + r")-([a-z]+)\b"
    return re.sub(pattern, lambda m: m.group(1) + "_" + m.group(2), text)

# 3. Self- word protection
def protect_self_terms(text):
    self_terms = set(re.findall(r"\bself[\s\-]*[a-z]+\b", text))
    for term in self_terms:
        token = re.sub(r"[\s\-]+", "_", term)
        text = re.sub(re.escape(term), token, text)
    return text, self_terms

# 4. Self- word restoration
def restore_self_terms(text, self_terms):
    for term in self_terms:
        token = re.sub(r"[\s\-]+", "_", term)
        text = text.replace(token, term)
    return text

# 5. Auto-fix split words
def auto_fix_split_words(text):
    tokens = text.split()
    fixed_tokens = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            combined = tokens[i] + tokens[i + 1]
            if dictionary.check(combined):
                fixed_tokens.append(combined)
                i += 2
                continue
        fixed_tokens.append(tokens[i])
        i += 1
    return " ".join(fixed_tokens)

# 6. Preprocessing function for SBERT
def preprocess_for_sbert(text, min_len=6, max_len=300):
    text = text.lower()
    text = fix_line_breaks(text)
    text = protect_meaningful_hyphen_words(text)                    

    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text, flags=re.DOTALL)

    # Standardize medical terms
    text = re.sub(r"hba\s*1c", "a1c", text, flags=re.IGNORECASE)
    text = re.sub(r"\ba\s*1c\b", "a1c", text, flags=re.IGNORECASE)
    text = re.sub(r"\ba1c\b", "aic", text)
    text = re.sub(r"\b(t2dm|t2d)\b", "ttdm", text, flags=re.IGNORECASE)
    text = re.sub(r"\btype[-_\s]*[2two]+\s*diabetes([-_\s]*mellitus)?\b", "ttdm", text, flags=re.IGNORECASE)

    text, self_terms = protect_self_terms(text)
    text = re.sub(r"http\S+|www\S+", "", text)                       
    text = re.sub(r"[^a-zA-Z0-9.,!?_\- ]", " ", text)                
    text = re.sub(r"\s+", " ", text).strip()                       
    text = auto_fix_split_words(text)

    text = text.replace("aic", "a1c")
    text = text.replace("ttdm", "t2dm")
    text = restore_self_terms(text, self_terms)

    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if min_len <= len(sent.text.split()) <= max_len]

# 7. Process Markdown files and save cleaned text
markdown_folder = r"C:____________________"
cleaned_folder = os.path.join(markdown_folder, "Cleaned")
pathlib.Path(cleaned_folder).mkdir(parents=True, exist_ok=True)

for filename in os.listdir(markdown_folder):
    if filename.endswith(".md"):
        input_path = os.path.join(markdown_folder, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(cleaned_folder, base_name + ".txt")

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned_sentences = preprocess_for_sbert(raw_text)

            with open(output_path, "w", encoding="utf-8") as f_out:
                for sent in cleaned_sentences:
                    f_out.write(sent + "\n")

            print(f"Completed: {base_name}.txt (Sentence Count: {len(cleaned_sentences)})")

        except Exception as e:
            print(f"Error: {filename} | Reason: {e}")

