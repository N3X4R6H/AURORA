from pathlib import Path
from typing import Iterable, List, Optional
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


class AuroraTokenizer:
    """Byte-Level BPE Tokenizer: guarantees 100% lossless encoding and preserves indentation."""

    SPECIAL_TOKENS = [
        "<|endoftext|>",  # Marks file boundary
        "<|pad|>",         # Padding token for uniform batch size
        "<|fim_prefix|>",  # Fill-in-the-middle prefix
        "<|fim_middle|>",  # Fill-in-the-middle middle
        "<|fim_suffix|>",  # Fill-in-the-middle suffix
    ]

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.BPE())
        # Use ByteLevel to map raw bytes and preserve leading/indentation spaces
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    def train_from_iterator(self, iterator: Iterable[str], min_frequency: int = 2) -> None:
        """Trains BPE merges from an in-memory iterable of code strings."""
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        self.tokenizer.train_from_iterator(iterator, trainer=trainer)

    def train_from_files(self, file_paths: List[str], min_frequency: int = 2) -> None:
        """Trains BPE merges directly from a list of code file paths on disk."""
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        self.tokenizer.train(files=file_paths, trainer=trainer)

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        """Converts raw text into token IDs, optionally appending <|endoftext|>."""
        if not text:
            return []
        ids = self.tokenizer.encode(text).ids
        if add_eos:
            eos_id = self.token_to_id("<|endoftext|>")
            if eos_id is not None:
                ids.append(eos_id)
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Converts token IDs back into text, preserving exact whitespace and symbols."""
        if not token_ids:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        """Returns the integer ID for a token string, or None if not found."""
        return self.tokenizer.token_to_id(token)

    def calculate_efficiency(self, text: str) -> float:
        """Returns compression efficiency: characters per token (len(text) / len(tokens))."""
        if not text:
            return 0.0
        ids = self.encode(text)
        return len(text) / len(ids) if ids else 0.0

    def save(self, path: str = "tokenizer.json") -> None:
        """Saves trained vocabulary and merge rules to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str = "tokenizer.json") -> "AuroraTokenizer":
        """Loads a pre-trained tokenizer JSON file from disk."""
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(path)
        instance.vocab_size = instance.tokenizer.get_vocab_size()
        return instance

