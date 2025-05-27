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

HEADER_INFO = {
    "gpt-2": {
        "magic": 20240520,
        "version": 1,
        "token_dtype": np.uint16,
    },
}

def write_datafile(filename, toks, model_name="gpt-2"):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int16s
    - The tokens follow as uint16 (gpt-2)
    """
    assert (0 <= len(toks)).all() and (len(toks) < 2**16).all(), "token dictionary too large for uint16"
    assert model_name in ["gpt-2"], "Only gpt-2 is supported for now"
    header_info = HEADER_INFO[model_name]
    ## header construct
    header = np.zeros(256, dtype=header_info["token_dtype"])
    header[0] = header_info["magic"]
    header[1] = header_info["version"]
    header[2] = len(toks)
    ## data construct
    data = np.array(toks, dtype=header_info["token_dtype"])
    ## write
    with open(filenam) as f:
        f.write(header.tobytes())
        f.write(data.tobytes())

def tokenize(doc):
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    tokens = [eot]
    tokens.extend(enc.encode(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    token_np_unit = tokens_np.astype(np.uint16)
    return token_np_unit

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])
