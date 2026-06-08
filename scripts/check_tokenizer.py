from __future__ import annotations

from transformers import AutoConfig

from vn_summarization.modeling import _load_tokenizer


def main() -> None:
    model_name = "VietAI/vit5-base"
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = _load_tokenizer(
        name_or_path=model_name,
        model_config=config,
        model_cfg={"use_fast_tokenizer": False},
        trust_remote_code=False,
        cache_dir=None,
    )
    encoded = tokenizer("summarize: Đây là một bài kiểm tra tokenizer tiếng Việt.")
    print(type(tokenizer).__name__)
    print("is_fast", getattr(tokenizer, "is_fast", False))
    print("vocab_size", len(tokenizer))
    print("pad/eos/unk", tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id)
    print("tokens", len(encoded["input_ids"]))


if __name__ == "__main__":
    main()
