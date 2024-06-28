"""script to load vocab and merges to test the tokenizer"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tokenizer import AdvancedTokenizer 

vocab_size = 1_000

if __name__ == "__main__":
    tokenizer = AdvancedTokenizer(vocab_size=vocab_size, min_frequency=2, special_tokens=["<|endoftext|>"])
    
    # loading the attributes previously trained
    tokenizer_name = 'adv_tokenizer'
    tokenizer = tokenizer.load(path=f'src/attributes/{tokenizer_name}.json')   
    
    test_text = """
    import torch

    class FixedCategorical(torch.distributions.Categorical):
        def sample(self):
            return super().sample().unsqueeze(-1)
        
        def log_probs(self, actions):
            return (
                super()
                .log_prob(actions.squeeze(-1))
                .view(actions.size(0), -1)
                .sum(-1)
                .unsqueeze(-1)
            )
        
        def mode(self):
            return self.probs.argmax(dim=-1, keepdim=True)
    """#"hey everyone, this test is tokenized using an advanced tokenizer from scratch. fineweb edu is the training dataset."
    encoded = tokenizer.encode(test_text)
    print('\nencoded: ', encoded)
    
    decoded = tokenizer.decode(encoded)
    print('\ndecoded: ', decoded)
    
    tokens = tokenizer.tokenize(test_text)
    print('\ntokens length:', len(tokens))
    # doing preprocessing on tokens to print and don't see that weird Ġ
    print('\ntokens:', [token.replace('Ġ', ' ') for token in tokens])