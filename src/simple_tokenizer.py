"""tokenizer from scratch"""
import json
from pathlib import Path

class SimpleTokenizer:
    def __init__(self, vocab_size=300):
        """
        Initialize the SimpleTokenizer object.

        Args:
            vocab_size (int): The size of the vocabulary. Default is 300.

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

    def encode(self, text):
        """
        Encodes the given text into tokens using the tokenizer's merge algorithm.

        Args:
            text (str): The input text to be encoded.

        Returns:
            list: The list of tokens obtained after encoding the text.
        """
        tokens = list(text.encode("utf-8"))
        
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break  
            
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
    
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
    
    def _save_vocab_file(self, filename, root_path):
        """
        Save the tokenizer's vocabulary to a JSON file.

        Args:
            filename (str): The name of the file to save the vocabulary to.
            root_path (str): The root path where the file will be saved.

        Returns:
            None
        """
        serializable_vocab = {}
    
        for k, v in self.vocab.items():
            try:
                decoded_str = v.decode("utf-8")
                serializable_vocab[k] = decoded_str
            except UnicodeDecodeError:
                serializable_vocab[k] = list(v) 
        
        with open(Path(root_path)/filename, "w") as f:
            json.dump(serializable_vocab, f)
    
    def _save_merge_file(self, filename, root_path):
        """
        Save the tokenizer's merges to a file.

        Args:
            filename (str): The name of the file to save the merges to.
            root_path (str): The root path where the file will be saved.

        Returns:
            None
        """
        with open(Path(root_path)/filename, "w") as f:
            for (a, b), idx in self.merges.items():
                f.write(f"{a}\t{b}\t{idx}\n")

    def save_attributes(self, vocab_filename, merges_filename, root_path):
        """
        Save the tokenizer's vocabulary and merges to files.

        Args:
            vocab_filename (str): The name of the file to save the vocabulary to.
            merges_filename (str): The name of the file to save the merges to.
            root_path (str): The root path where the files will be saved.

        Returns:
            None
        """
        self._save_vocab_file(vocab_filename, root_path)
        self._save_merge_file(merges_filename, root_path)
        
    def _load_vocab_file(self, filename, root_path):
        """
        Load the tokenizer's vocabulary from a JSON file.

        Args:
            filename (str): The name of the file containing the vocabulary.
            root_path (str): The root path where the file is located.

        Returns:
            dict: The loaded vocabulary as a dictionary mapping integer IDs to bytes or strings.
        """
        vocab_path = Path(root_path) / filename
        with open(vocab_path, "r") as f:
            vocab_data = json.load(f)
        
        loaded_vocab = {int(k): v.encode("utf-8") if isinstance(v, str) else bytes(v) for k, v in vocab_data.items()}
        return loaded_vocab
    
    def _load_merge_file(self, filename, root_path):
        """
        Load the tokenizer's merges from a text file.

        Args:
            filename (str): The name of the file containing the merges.
            root_path (str): The root path where the file is located.

        Returns:
            dict: The loaded merges as a dictionary mapping tuple of integers to integers.
        """
        merges_path = Path(root_path) / filename
        merges = {}
        with open(merges_path, "r") as f:
            for line in f:
                a, b, idx = line.strip().split("\t")
                merges[(int(a), int(b))] = int(idx)
        
        return merges
    
    def load_attributes(self, vocab_filename, merges_filename, root_path):
        """
        Load the tokenizer's vocabulary and merges from files.

        Args:
            vocab_filename (str): The name of the file containing the vocabulary.
            merges_filename (str): The name of the file containing the merges.
            root_path (str): The root path where the files are located.

        Returns:
            None
        """
        self.vocab = self._load_vocab_file(vocab_filename, root_path)
        self.merges = self._load_merge_file(merges_filename, root_path)