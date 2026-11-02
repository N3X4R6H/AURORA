import pytest
from aurora.data.tokenizer import AuroraTokenizer
from aurora.data.packing import SequenceBuilder


@pytest.fixture(scope="module")
def trained_tokenizer():
    corpus = [
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        "class Node:\n    def __init__(self, val: int):\n        self.val = val\n",
        "for i in range(10):\n    if i % 2 == 0:\n        print(f'Number: {i}')\n",
        "# Unicode: π ≈ 3.14159 | testing 🙂\nx = '🙂'\n",
    ] * 20
    tok = AuroraTokenizer(vocab_size=500)
    tok.train_from_iterator(corpus)
    return tok


def test_empty_input(trained_tokenizer):
    assert trained_tokenizer.encode("") == []
    assert trained_tokenizer.decode([]) == ""
    assert trained_tokenizer.calculate_efficiency("") == 0.0


def test_special_tokens(trained_tokenizer):
    eos_id = trained_tokenizer.token_to_id("<|endoftext|>")
    pad_id = trained_tokenizer.token_to_id("<|pad|>")
    assert eos_id is not None
    assert pad_id is not None
    assert eos_id != pad_id


def test_lossless_roundtrip_edge_cases(trained_tokenizer):
    test_cases = [
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        "   \t\t \n \n\t",
        "if (n := len(data)) > 10:\n    matrix = a @ b ** 2 // 3\n",
        "# testing 🙂 | Math: α + β = π | Russian: привет",
        "def ??? {{{ [[[[ @@@ \\\n    return ;;; `~!",
        "if True:\n    if True:\n        if True:\n            if True:\n                pass",
    ]
    for text in test_cases:
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == text, f"Lossless check failed on: {text!r}"


def test_serialization(trained_tokenizer, tmp_path):
    save_file = str(tmp_path / "test_tok.json")
    trained_tokenizer.save(save_file)

    reloaded = AuroraTokenizer.load(save_file)
    sample = "def test():\n    return 42\n"
    assert reloaded.decode(reloaded.encode(sample)) == sample
    assert reloaded.tokenizer.get_vocab_size() == trained_tokenizer.tokenizer.get_vocab_size()


def test_sequence_builder_packing(trained_tokenizer):
    builder = SequenceBuilder(trained_tokenizer, context_length=32)

    sample_1 = "def f(x):\n    return x + 1\n"
    sample_2 = "def g(y):\n    return y * 2\n"

    chunks = builder.pack_sequences([sample_1, sample_2], drop_last=False)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk) == 32

    # Check that EOS and PAD are appropriately present
    assert builder.eos_id in chunks[0]

