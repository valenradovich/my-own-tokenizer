"""script to load vocab and merges to test the tokenizer"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.base_tokenizer import BaseTokenizer

if __name__ == "__main__":
    tokenizer = BaseTokenizer(vocab_size=400)
    
    # loading the attributes previously trained
    tokenizer.load_attributes(vocab_filename='vocab.json', merges_filename='merges.txt', root_path='src/attributes/')   

    # example
    encoded = tokenizer.encode("""hey everyone, this test is tokenized using a simple tokenizer from scratch. fineweb edu is the training dataset.""")
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)