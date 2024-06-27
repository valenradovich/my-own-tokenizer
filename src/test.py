"""script to load vocab and merges to test the tokenizer"""
from simple_tokenizer import SimpleTokenizer

if __name__ == "__main__":
    tokenizer = SimpleTokenizer(vocab_size=400)
    
    # loading the attributes previously trained
    tokenizer.load_attributes(vocab_filename='vocab.json', merges_filename='merges.txt', root_path='src/attributes/')   

    # example
    encoded = tokenizer.encode("""hey everyone, this test is tokenized using a simple tokenizer from scratch. fineweb edu is the training dataset.""")
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)