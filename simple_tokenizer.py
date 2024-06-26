
class SimpleTokenizer:
    def __init__(self, vocab_size=276):
        """
        Initialize the SimpleTokenizer object.

        Args:
            vocab_size (int): The size of the vocabulary. Default is 276.

        Attributes:
            vocab_size (int): The size of the vocabulary.
            num_merges (int): The number of merges.
            merges (dict): A dictionary that maps a tuple of two integers to an integer.
            vocab (dict): A dictionary that maps an integer to a byte.

        """
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256
        self.merges = {}  # (int, int) -> int
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
    
    def get_stats(self, ids):
        """
        Calculate the frequency of each pair of consecutive elements in the given list.

        Args:
            ids (list): A list of elements.

        Returns:
            dict: A dictionary where the keys are pairs of consecutive elements and the values are their frequencies.
        """
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        """
        Merge consecutive pairs of elements in the given list of ids.

        Args:
            ids (list): The list of ids to be merged.
            pair (tuple): The pair of elements to be merged.
            idx (int): The index to be inserted in place of the merged pair.

        Returns:
            list: The new list of ids after merging.

        """
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text):
        """
        Trains the tokenizer by merging tokens based on the given text.

        Args:
            text (str): The text used for training the tokenizer.

        Returns:
            None
        """
        ids = list(text.encode("utf-8"))
        for i in range(self.num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merging {pair} into a new token {idx}")
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def decode(self, ids):
        """
        Decodes a list of token IDs into a text string.

        Args:
            ids (list): A list of token IDs.

        Returns:
            str: The decoded text string.
        """
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        """
        Encodes the given text into tokens using the tokenizer's merge algorithm.

        Args:
            text (str): The input text to be encoded.

        Returns:
            list: The list of tokens obtained after encoding the text.
        """
        # print(f"final merges = {self.merges}")
        
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
