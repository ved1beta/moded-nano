import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
import tiktoken
import numpy as np
from nltk.corpus import stopwords
from tokenizers.normalizers import NFD, StripAccents
from datasets import load_dataset
from tokenizers import Tokenizer
import multiprocessing as mp
import time

fw = load_dataset('rotten_tomatoes', split="train[:10]")

def process_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filterd_tokens = [word for word in tokens if word not in stop_words]
    clean_text = ' '.join(filterd_tokens)

    return clean_text

def tokenize(doc):
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    tokens = [eot]
    tokens.extend(enc.encode(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    token_np_unit = tokens_np.astype(np.uint16)
    return token_np_unit

def process_batch(batch):
    return [process_text(item['text']) for item in batch]

def tokenize_batch(batch):
    return [tokenize(item) for item in batch]

if __name__ == '__main__':
    # Process texts in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        cleaned_dataset = pool.map(process_text, [item['text'] for item in fw])

    print("Processed texts:")
    for i, text in enumerate(cleaned_dataset):
        print(f"\nText {i+1}:")
        print(text)

    # Tokenize in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        tokenized_dataset = pool.map(tokenize, [{'text': text} for text in cleaned_dataset])

    print("Tokenized texts:")
    for i, tokens in enumerate(tokenized_dataset):
        print(f"\nTokens for Text {i+1}:")
        print(tokens)
