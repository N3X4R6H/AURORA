"""CLI demonstration and verification for the AURORA Byte-Level BPE Tokenizer."""

import sys
from pathlib import Path

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurora.data.tokenizer import AuroraTokenizer
from aurora.data.packing import SequenceBuilder


def main():
    corpus = [
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        "class Model:\n    def __init__(self):\n        pass\n",
    ] * 20

    # 1. Train and save
    tokenizer = AuroraTokenizer(vocab_size=500)
    tokenizer.train_from_iterator(corpus)
    tokenizer.save("tokenizer.json")

    # 2. Load
    tokenizer = AuroraTokenizer.load("tokenizer.json")

    # 3. Check special token IDs
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    print(f"Special IDs -> EOS: {eos_id}, PAD: {pad_id}")

    # 4. Encode, decode, and efficiency
    sample_code = "def add(a, b):\n    return a + b\n"
    tokens = tokenizer.encode(sample_code, add_eos=True)
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    efficiency = tokenizer.calculate_efficiency(sample_code)

    print("Original:\n", sample_code.strip())
    print("Tokens:", tokens)
    print("Decoded:\n", decoded.strip())
    print(f"Lossless: {decoded == sample_code}")
    print(f"Efficiency (chars/token): {efficiency:.2f}")

    # 5. Sequence builder (normalize, sample_to_tokens, pack_sequences)
    builder = SequenceBuilder(tokenizer, context_length=32)

    normalized = builder.normalize_code("x = 1\r\ny = 2\r\n")
    sample_tokens = builder.sample_to_tokens(sample_code)
    chunks = builder.pack_sequences([sample_code, normalized], drop_last=False)

    print(f"Sample tokens (with EOS): {sample_tokens}")
    print(f"Packed into {len(chunks)} chunk(s) of length {len(chunks[0])}:")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")


if __name__ == "__main__":
    main()
