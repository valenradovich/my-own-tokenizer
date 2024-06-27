"""script to train the tokenizer from scratch using a dataset"""
from datasets import load_dataset, Dataset
from simple_tokenizer import SimpleTokenizer

### choose the dataset to train the tokenizer you want to use, take care of vocab_size as well
# dataset = Dataset.load_from_disk('data/fineweb_edu_sample_250k')
dataset = Dataset.load_from_disk('data/fineweb_edu_sample_0.5k')
#dataset = Dataset.load_from_disk('data/fineweb_edu_sample_50')
print('dataset loaded:', len(dataset))

training_text = "\n".join(dataset['text'])
print('training text done:', len(training_text))

vocab_size = 1_000

if __name__ == "__main__":
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    
    tokenizer.train(training_text)
    
    # now you can try the tokenizer to encode and decode text
    encoded = tokenizer.encode("""hey everyone, this test is tokenized using a simple tokenizer from scratch. fineweb edu is the training dataset.""")
    print(encoded)
    
    decoded = tokenizer.decode(encoded)
    print(decoded)
    
    tokenizer.save_attributes(vocab_filename='vocab.json', merges_filename='merges.txt', root_path='src/attributes/')
    print('attributes saved')