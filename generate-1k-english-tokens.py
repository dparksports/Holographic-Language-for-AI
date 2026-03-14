from transformers import AutoTokenizer

def generate_1000_english_tokens():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    vocab = tokenizer.get_vocab()
    # Sort by ID (lower IDs are the core, high-frequency base words)
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    standard_words = []
    for word, token_id in sorted_vocab:
        clean_word = word.replace('Ġ', '').strip() # Remove BPE byte-space markers
        # Filter for pure alphabetical words longer than 3 characters
        if clean_word.isalpha() and clean_word.islower() and len(clean_word) > 3:
            standard_words.append(clean_word)
        if len(standard_words) == 1000: break
    return standard_words