import pytest


@pytest.mark.skip(reason="Enable once model, tokenizer and trainer exist")
def test_end_to_end_smoke_and_checkpoint_resume():
    """Planned: tiny synthetic corpus -> few steps -> save checkpoint -> resume
    -> loss continues from the same step with identical optimizer state."""
