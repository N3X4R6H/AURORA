from typing import Iterable, List
from aurora.data.tokenizer import AuroraTokenizer


class SequenceBuilder:
    """Slices tokenized code files into fixed-length chunks for Transformer training."""

    def __init__(self, tokenizer: AuroraTokenizer, context_length: int = 512):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.eos_id = tokenizer.token_to_id("<|endoftext|>")
        self.pad_id = tokenizer.token_to_id("<|pad|>")

    @staticmethod
    def normalize_code(code: str) -> str:
        """Converts Windows (CRLF) and Mac (CR) line endings into standard Unix (LF)."""
        return code.replace("\r\n", "\n").replace("\r", "\n")

    def sample_to_tokens(self, code: str) -> List[int]:
        """Normalizes code, encodes to token IDs, and appends <|endoftext|>."""
        if not code:
            return []
        clean = self.normalize_code(code)
        return self.tokenizer.encode(clean, add_eos=True)

    def pack_sequences(self, code_samples: Iterable[str], drop_last: bool = False) -> List[List[int]]:
        """Packs multiple code samples into fixed context_length chunks; pads leftovers."""
        stream: List[int] = []
        chunks: List[List[int]] = []

        for sample in code_samples:
            if sample:
                stream.extend(self.sample_to_tokens(sample))

        # Slice stream into exact context_length blocks
        while len(stream) >= self.context_length:
            chunks.append(stream[: self.context_length])
            stream = stream[self.context_length :]

        # Pad remaining tokens with <|pad|> if not dropped
        if stream and not drop_last:
            pad_count = self.context_length - len(stream)
            chunks.append(stream + [self.pad_id] * pad_count)

        return chunks

