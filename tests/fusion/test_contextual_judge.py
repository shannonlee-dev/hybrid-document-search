"""Exercise the real judge adapter with controlled tokenizer/model boundaries."""

from math import log
from types import SimpleNamespace

import pytest

from fusion.contextual_judge import PREFIX, SUFFIX, LocalRelevanceJudge

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


class _Tokenizer:
    unk_token_id = 0

    def __init__(self, token_ids=None):
        self.token_ids = token_ids or {"no": 3, "yes": 1}

    def convert_tokens_to_ids(self, token):
        return self.token_ids[token]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == PREFIX:
            return [2, 2]
        if text == SUFFIX:
            return [6, 6]
        return [7, 8] if text.endswith("dense body") else [9, 10, 11, 12]

    def pad(self, sequences, *, padding, return_tensors):
        assert padding is True and return_tensors == "pt"
        rows = [s["input_ids"] for s in sequences]
        length = max(map(len, rows))
        return {
            "input_ids": torch.tensor([[0] * (length - len(r)) + r for r in rows]),
            "attention_mask": torch.tensor(
                [[0] * (length - len(r)) + [1] * len(r) for r in rows]
            ),
        }


class _FullLogitsModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask, *, use_cache):
        assert not use_cache and not torch.is_grad_enabled() and not self.training
        assert input_ids.device.type == "cpu"
        assert input_ids.tolist() == [[0, 2, 2, 7, 8, 6, 6], [2, 2, 9, 10, 11, 6, 6]]
        assert attention_mask.tolist() == [[0, 1, 1, 1, 1, 1, 1], [1] * 7]
        # Earlier positions and unrelated vocabulary tokens deliberately disagree.
        logits = torch.zeros(2, 7, 13)
        logits[:, :-1, 3] = 20
        logits[:, -1, 12] = 100
        logits[0, -1, 1] = log(3)
        logits[1, -1, 3] = log(4)
        return SimpleNamespace(logits=logits)


class _LastLogitsModel(_FullLogitsModel):
    def forward(self, input_ids, attention_mask, *, use_cache, logits_to_keep):
        assert logits_to_keep == 1
        output = super().forward(input_ids, attention_mask, use_cache=use_cache)
        return SimpleNamespace(logits=output.logits[:, -1:, :])


@pytest.fixture
def make_judge(monkeypatch):
    def create(model_type=_FullLogitsModel, *, token_ids=None):
        def load_tokenizer(path, *, padding_side, local_files_only):
            assert path == "local-snapshot" and padding_side == "left"
            assert local_files_only is True
            return _Tokenizer(token_ids)

        def load_model(path, *, dtype, attn_implementation, local_files_only):
            assert path == "local-snapshot" and dtype == torch.float32
            assert attn_implementation == "sdpa" and local_files_only is True
            return model_type()

        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", load_tokenizer
        )
        monkeypatch.setattr(
            transformers.AutoModelForCausalLM, "from_pretrained", load_model
        )
        return LocalRelevanceJudge("local-snapshot", device="cpu", max_tokens=7)

    return create


@pytest.mark.parametrize("model_type", [_FullLogitsModel, _LastLogitsModel])
def test_judge_returns_yes_probability_from_last_position_in_dense_sparse_order(
    make_judge, model_type
):
    judge = make_judge(model_type)
    for invocation in (1, 2):
        probabilities = judge("question", ["dense body", "sparse body"])
        assert probabilities == pytest.approx([0.75, 0.2])
        assert all(type(value) is float for value in probabilities)
        assert judge.pairs == 2 * invocation
        assert judge.truncated_pairs == invocation


@pytest.mark.parametrize("token_ids", [{"no": 1, "yes": 1}, {"no": 0, "yes": 1}])
def test_judge_rejects_ambiguous_or_unknown_yes_no_tokens(make_judge, token_ids):
    with pytest.raises(ValueError, match="distinct yes/no tokens"):
        make_judge(token_ids=token_ids)


@pytest.mark.parametrize("texts", [[], ["dense body"], ["a", "b", "c"]])
def test_judge_requires_exactly_two_bodies_before_forward(make_judge, texts):
    judge = make_judge()
    with pytest.raises(ValueError, match="Dense and Sparse top1"):
        judge("question", texts)
    assert judge.pairs == judge.truncated_pairs == 0
