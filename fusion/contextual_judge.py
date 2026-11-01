"""Local Qwen relevance judgement for two top1 bodies; no labels or APIs."""

import inspect

PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def judge_inputs(tokenizer, query, texts, *, max_tokens=2048):
    """Truncate the prompt body while preserving system and answer suffix tokens."""
    pre = tokenizer.encode(PREFIX, add_special_tokens=False)
    post = tokenizer.encode(SUFFIX, add_special_tokens=False)
    budget = max_tokens - len(pre) - len(post)
    if budget <= 0:
        raise ValueError("judge token budget cannot hold system and answer suffix")
    sequences, truncated = [], 0
    for text in texts:
        body = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {text}"
        tokens = tokenizer.encode(body, add_special_tokens=False)
        truncated += len(tokens) > budget
        sequences.append({"input_ids": pre + tokens[:budget] + post})
    return sequences, truncated


class LocalRelevanceJudge:
    """Load a supplied local snapshot and evaluate Dense then Sparse top1 bodies."""

    def __init__(self, model_path, *, device="cuda", max_tokens=2048):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device, self.max_tokens = device, max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            local_files_only=True,
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        self.no, self.yes = [
            self.tokenizer.convert_tokens_to_ids(t) for t in ("no", "yes")
        ]
        if self.no == self.yes or self.tokenizer.unk_token_id in (self.no, self.yes):
            raise ValueError("judge tokenizer must contain distinct yes/no tokens")
        self.extra = (
            {"logits_to_keep": 1}
            if "logits_to_keep" in inspect.signature(self.model.forward).parameters
            else {}
        )
        self.pairs = self.truncated_pairs = 0

    def __call__(self, query, texts):
        import torch

        if len(texts) != 2:
            raise ValueError("judge requires Dense and Sparse top1 texts")
        sequences, truncated = judge_inputs(
            self.tokenizer, query, texts, max_tokens=self.max_tokens
        )
        features = self.tokenizer.pad(sequences, padding=True, return_tensors="pt")
        features = {k: v.to(self.device) for k, v in features.items()}
        with torch.inference_mode():
            logits = self.model(**features, use_cache=False, **self.extra).logits
            probabilities = logits[:, -1, [self.no, self.yes]].float().softmax(-1)[:, 1]
        self.pairs += 2
        self.truncated_pairs += truncated
        return probabilities.cpu().numpy().astype(float).tolist()
