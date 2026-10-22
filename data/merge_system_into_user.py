"""Merge system prompt into user prompt, drop the system message, and filter
samples whose chat-templated prompt exceeds 1024 tokens.

The chat template is defined in the Qwen2.5-Math-1.5B-16k-think
tokenizer_config.json.
"""
import argparse
import os

TOKENIZER_PATH = os.environ.get("LUFFY_TOKENIZER_PATH")
MAX_PROMPT_TOKENS = 1024
BATCH_SIZE = 256


def main():
    parser = argparse.ArgumentParser(description="Merge system messages into user prompts and filter by token count")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH, required=not TOKENIZER_PATH,
                        help="Local tokenizer directory (or set LUFFY_TOKENIZER_PATH)")
    args = parser.parse_args()
    if not os.path.isdir(args.tokenizer):
        raise FileNotFoundError(f"Local tokenizer directory not found: {args.tokenizer}")
    import pandas as pd
    from transformers import AutoTokenizer

    parquet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openr1-code.parquet')
    df = pd.read_parquet(parquet_path)
    total = len(df)
    print(f"Total samples: {total}", flush=True)

    # 1. Merge system into user, drop system
    def merge_prompt(messages):
        system_content = ""
        user_content = ""
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            elif m["role"] == "user":
                user_content = m["content"]
        merged = (system_content + "\n\n" + user_content).strip() if system_content else user_content
        return [{"role": "user", "content": merged}]

    df["prompt"] = df["prompt"].map(merge_prompt)

    # Verify
    for m in df.iloc[0]["prompt"]:
        print(f"  role={m['role']}, len={len(m['content'])}")
    print(f"Sample merged user prompt (first 200 chars): {df.iloc[0]['prompt'][0]['content'][:200]}", flush=True)

    # 2. Load tokenizer and count tokens with chat template
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)

    prompt_texts = df["prompt"].tolist()
    token_counts = []

    for i in range(0, len(prompt_texts), BATCH_SIZE):
        batch = prompt_texts[i : i + BATCH_SIZE]
        for chat in batch:
            text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            token_counts.append(len(ids))
        done = min(i + BATCH_SIZE, len(prompt_texts))
        if done % (BATCH_SIZE * 20) < BATCH_SIZE:
            print(f"  tokenized {done}/{len(prompt_texts)}", flush=True)

    df["prompt_tokens_templated"] = token_counts

    # 3. Filter
    keep = df["prompt_tokens_templated"] <= MAX_PROMPT_TOKENS
    n_removed = int((~keep).sum())
    df_filtered = df[keep].reset_index(drop=True)

    print(f"\nPrompt token stats (with chat template):")
    print(f"  max={df['prompt_tokens_templated'].max()}, "
          f"mean={df['prompt_tokens_templated'].mean():.0f}, "
          f"median={df['prompt_tokens_templated'].median():.0f}")
    print(f"Samples removed (> {MAX_PROMPT_TOKENS} tokens): {n_removed}")
    print(f"Samples kept: {len(df_filtered)}")

    # Update prompt_tokens column to reflect the new templated count
    df_filtered["prompt_tokens"] = df_filtered["prompt_tokens_templated"]
    df_filtered = df_filtered.drop(columns=["prompt_tokens_templated"])

    df_filtered.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df_filtered)} samples to {parquet_path}", flush=True)


if __name__ == "__main__":
    main()
