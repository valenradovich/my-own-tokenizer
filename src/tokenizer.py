"""this tokenizer would use sophisticated techniques"""
from src.base_tokenizer import BaseTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.normalizers import Lowercase
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from typing import List, Iterator

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class AdvancedTokenizer(BaseTokenizer):
    def __init__(self, vocab_size=25000, min_frequency=2, special_tokens=None):
        super().__init__(vocab_size)
        self.min_frequency = min_frequency # minimum frequency of a token to be included in the vocabulary
        self.special_tokens = special_tokens or ["<|endoftext|>"] # special tokens to be included in the vocabulary
        self.tokenizer = Tokenizer(BPE())
        self.normalizer = Lowercase() # lowercase the text bc the tokenizer is not handling correctly via pattern
        self.tokenizer.pre_tokenizer = Sequence([
            Split(pattern=GPT4_SPLIT_PATTERN, behavior="removed", invert=False),
            ByteLevel(add_prefix_space=False)
        ])
        self.tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
        self.tokenizer.decoder = ByteLevelDecoder()

    def train(self, texts: Iterator[str]):
        """
        Train the tokenizer on the given texts.

        Args:
            texts (Iterator[str]): An iterator of text strings to train on.
        """
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens
        )
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

    def encode(self, text: str) -> List[int]:
        """
        Encode the given text into a list of token ids.

        Args:
            text (str): The text to encode.

        Returns:
            List[int]: A list of token ids.
        """
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode the given list of token ids into text.

        Args:
            ids (List[int]): A list of token ids to decode.

        Returns:
            str: The decoded text.
        """
        return self.tokenizer.decode(ids)#, skip_special_tokens=False)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the given text into a list of tokens.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of tokens.
        """
        return self.tokenizer.encode(text).tokens

    def save(self, path: str):
        """
        Save the tokenizer to a file.

        Args:
            path (str): The path where the tokenizer should be saved.
        """
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str):
        """
        Load a tokenizer from a file.

        Args:
            path (str): The path from which to load the tokenizer.

        Returns:
            AdvancedTokenizer: The loaded tokenizer.
        """
        tokenizer = cls()
        tokenizer.tokenizer = Tokenizer.from_file(path)
        return tokenizer

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a batch of texts.

        Args:
            texts (List[str]): A list of texts to encode.

        Returns:
            List[List[int]]: A list of lists of token ids.
        """
        return [self.encode(text) for text in texts]

    def batch_decode(self, batch_ids: List[List[int]]) -> List[str]:
        """
        Decode a batch of token id lists.

        Args:
            batch_ids (List[List[int]]): A list of lists of token ids to decode.

        Returns:
            List[str]: A list of decoded texts.
        """
        return [self.decode(ids) for ids in batch_ids]