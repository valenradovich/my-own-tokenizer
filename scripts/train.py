"""script to train the advanced tokenizer from scratch using a fineweb edu dataset. as you may notice is extremely fast compared to the base tokenizer"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import Dataset
from src.tokenizer import AdvancedTokenizer

### choose the dataset to train the tokenizer you want to use, take care of vocab_size as well
dataset_name = 'python_code'#'fineweb_edu_sample_500k' # *_50 or *_250k, *_500k or *_1k
dataset = Dataset.load_from_disk(f'data/{dataset_name}') 
print(f'dataset "{dataset_name}" loaded:', len(dataset))
print('amount of words:', sum([len(text.split()) for text in dataset['text']]))

vocab_size = 3_125

if __name__ == "__main__":
    tokenizer = AdvancedTokenizer(vocab_size=vocab_size, min_frequency=2, special_tokens=["<|endoftext|>"])
    
    tokenizer.train(dataset['text'])
    
    test_text = """
    def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]

    # Test the function
    for i in range(10):
        print(f"Fibonacci({i}) = {fibonacci(i)}")
    """#"hey everyone, this test is tokenized using an advanced tokenizer from scratch. fineweb edu is the training dataset."
    
    
    # now you can try the tokenizer to encode and decode text
    encoded = tokenizer.encode(test_text)
    print('\nencoded: ', encoded)
    
    decoded = tokenizer.decode(encoded)
    print('\ndecoded: ', decoded)
    
    tokens = tokenizer.tokenize(test_text)
    # doing preprocessing on tokens to print and don't see that weird Ġ all the time
    print('\ntokens:', [token.replace('Ġ', ' ') for token in tokens])
    
    tokenizer_name = 'adv_tokenizer'
    tokenizer.save(f'src/attributes/{tokenizer_name}.json')
    print('attributes saved')