"""Convert ``livecodebench.jsonl`` into the ``openr1-code-valid.parquet`` schema.

Output columns (identical to ``openr1-code-valid.parquet``):

    data_source      'livecodebench'
    prompt           [{'role': 'user', 'content': <instruction + problem>}]
                     (single user message, system already merged, same
                     C++17 instruction template as the openr1-code data)
    target           [{'role': 'assistant', 'content': ''}]
                     (LiveCodeBench ships no reference solution)
    examples         [{'input': ..., 'output': ...}, ...]
                     parsed from ``public_test_cases`` -- ONLY the public
                     test cases; the ``testtype`` field is dropped.
    num_tokens / prompt_tokens / response_tokens   Qwen2.5 token counts

Rows whose ``public_test_cases`` is empty/unparseable (the functional
LeetCode problems with class-signature ``starter_code`` and no public
stdin tests) are dropped -- they cannot be scored by the local
stdin-driven C++ reward.

Usage:
    python prepare_livecodebench.py \
        --input ./livecodebench.jsonl \
        --output ./livecodebench.parquet \
        --tokenizer "$LUFFY_TOKENIZER_PATH"
"""

import argparse
import json
import os

INSTRUCTION = """You will be given a competitive programming problem.

Analyze the maximum input constraints and identify the optimal algorithmic approach and data structures needed to process the largest possible test cases within the time and memory limits, then explain why your chosen implementation strategy is the most efficient solution. Please reason step by step about your solution approach, then provide a complete implementation in C++17 that is thoroughly optimized for both speed and memory usage.

Your solution must read input from standard input (cin), write output to standard output (cout).
Do not include any debug prints or additional output.

Put your final solution within a single code block:
```cpp
<your code here>
```"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "livecodebench.jsonl"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "livecodebench.parquet"))
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("LUFFY_TOKENIZER_PATH"),
        help="Optional local tokenizer directory (or set LUFFY_TOKENIZER_PATH)",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def parse_public_test_cases(raw):
    """Parse the ``public_test_cases`` JSON string into [{input, output}]."""
    if not raw:
        return []
    try:
        cases = json.loads(raw)
    except (ValueError, TypeError):
        return []
    examples = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        if "input" not in case and "output" not in case:
            continue
        examples.append(
            {
                "input": "" if case.get("input") is None else str(case.get("input")),
                "output": "" if case.get("output") is None else str(case.get("output")),
            }
        )
    return examples


def count_prompt_tokens(tokenizer, chats, batch_size):
    counts = []
    for i in range(0, len(chats), batch_size):
        for chat in chats[i : i + batch_size]:
            text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            counts.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        if (i // batch_size) % 10 == 0:
            print(f"[tokenize] {min(i + batch_size, len(chats))}/{len(chats)}", flush=True)
    return counts


def main():
    args = parse_args()
    import pandas as pd

    records = []
    n_total = n_dropped = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            n_total += 1

            examples = parse_public_test_cases(item.get("public_test_cases"))
            if not examples:
                n_dropped += 1
                continue

            question = (item.get("question_content") or "").strip()
            starter = (item.get("starter_code") or "").strip()
            content = f"{INSTRUCTION}\n\n# Problem\n\n{question}"
            if starter:
                content += f"\n\n# Starter Code\n\n```\n{starter}\n```"

            records.append(
                {
                    "data_source": "livecodebench",
                    "prompt": [{"role": "user", "content": content}],
                    "target": [{"role": "assistant", "content": ""}],
                    "examples": examples,
                    "question_id": item.get("question_id"),
                }
            )

    print(f"[load] {n_total} rows, dropped {n_dropped} without public test cases, kept {len(records)}")
    if not records:
        raise RuntimeError("no sample survived the public-test-case filter")

    df = pd.DataFrame(records)

    tokenizer = None
    if args.tokenizer and os.path.isdir(args.tokenizer):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
        df["prompt_tokens"] = count_prompt_tokens(tokenizer, df["prompt"].tolist(), args.batch_size)
    else:
        print(f"[warn] tokenizer not found at {args.tokenizer!r}; token counts set to 0")
        df["prompt_tokens"] = 0
    df["response_tokens"] = 0
    df["num_tokens"] = df["prompt_tokens"] + df["response_tokens"]

    df = df.drop(columns=["question_id"])
    df = df[["data_source", "prompt", "target", "examples", "num_tokens", "prompt_tokens", "response_tokens"]]

    df.to_parquet(args.output, index=False)
    print(f"[save] {len(df)} rows -> {args.output}")
    print(
        "[stats] prompt_tokens:"
        f" max={df['prompt_tokens'].max()}, mean={df['prompt_tokens'].mean():.0f},"
        f" median={df['prompt_tokens'].median():.0f}"
    )
    print(
        "[stats] public test cases per row:"
        f" min={df['examples'].map(len).min()}, max={df['examples'].map(len).max()},"
        f" total={df['examples'].map(len).sum()}"
    )


if __name__ == "__main__":
    main()
