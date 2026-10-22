"""Preprocess the openr1-code parquet shards into a single training parquet file.

Steps:
  1. Read every ``*.parquet`` under ``--input_dir`` and concatenate them.
  2. Use the Qwen2.5 tokenizer to count the prompt (user) tokens and the
     response (assistant) tokens of every sample, then drop the samples whose
     prompt is longer than ``--max_prompt_tokens`` or whose response is longer
     than ``--max_response_tokens``.
  3. Extract the common prefix shared by all queries and use it as the system
     prompt: a ``{"role": "system"}`` dict is inserted at the position of the
     original ``user`` dict and the query keeps only its own (unique) part.
  4. Parse the ``## Examples`` section of the query into
     ``[{"input": ..., "output": ...}, ...]`` and store it in a new ``examples``
     column.
  5. Rename ``source`` -> ``data_source``, build the ``prompt`` column
     (system + user messages) and the ``target`` column (assistant message),
     and finally drop the original ``messages`` column.

Usage:
    python prepare_openr1_code.py \
        --input_dir ./openr1-code \
        --output ./openr1-code.parquet \
        --tokenizer "$LUFFY_TOKENIZER_PATH"
"""

import argparse
import glob
import os
import re


# ``` input ... ``` followed by ``` output ... ```
EXAMPLE_PATTERN = re.compile(
    r"```input\s*\n(.*?)\n?```\s*```output\s*\n(.*?)\n?```",
    re.DOTALL,
)
SECTION_PATTERN = re.compile(r"^##\s+\S", re.MULTILINE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=os.path.join(os.path.dirname(__file__), "openr1-code"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "openr1-code.parquet"))
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("LUFFY_TOKENIZER_PATH"),
        required=not os.environ.get("LUFFY_TOKENIZER_PATH"),
        help="Local tokenizer directory (or set LUFFY_TOKENIZER_PATH)",
    )
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)
    parser.add_argument("--max_response_tokens", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def load_parquet_dir(input_dir):
    import pandas as pd

    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet file found in {input_dir}")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        print(f"[load] {os.path.basename(f)}: {len(df)} rows")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def get_message(messages, role):
    """Return the content of the first message with the given role."""
    for msg in messages:
        if msg.get("role") == role:
            return msg["content"]
    return None


def count_tokens(tokenizer, texts, batch_size):
    counts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        counts.extend(len(ids) for ids in encoded)
        if (i // batch_size) % 50 == 0:
            print(f"[tokenize] {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return counts


def common_prefix(texts):
    """Longest common prefix of all texts, trimmed to a sentence/line boundary."""
    if not texts:
        return ""
    prefix = texts[0]
    for text in texts[1:]:
        upper = min(len(prefix), len(text))
        i = 0
        while i < upper and prefix[i] == text[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            return ""
    # avoid cutting in the middle of a word / sentence
    boundary = re.search(r"^.*[.!?\n]", prefix, re.DOTALL)
    return boundary.group(0) if boundary else prefix


def extract_examples(query):
    """Parse the ``## Examples`` section into a list of input/output dicts."""
    match = re.search(r"^##\s+Examples?\s*$", query, re.MULTILINE)
    if match is None:
        return []
    body = query[match.end() :]
    nxt = SECTION_PATTERN.search(body)
    if nxt is not None:
        body = body[: nxt.start()]
    return [
        {"input": inp, "output": out}
        for inp, out in EXAMPLE_PATTERN.findall(body)
    ]


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not glob.glob(os.path.join(args.input_dir, "*.parquet")):
        raise FileNotFoundError(f"no parquet file found in {args.input_dir}")
    if not os.path.isdir(args.tokenizer):
        raise FileNotFoundError(f"local tokenizer directory not found: {args.tokenizer}")
    from transformers import AutoTokenizer

    df = load_parquet_dir(args.input_dir)
    print(f"[load] total: {len(df)} rows, columns={df.columns.tolist()}")

    # ------------------------------------------------------------------ step 2
    df["query"] = df["messages"].map(lambda m: get_message(m, "user"))
    df["response"] = df["messages"].map(lambda m: get_message(m, "assistant"))
    df = df[df["query"].notna() & df["response"].notna()].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    df["prompt_tokens"] = count_tokens(tokenizer, df["query"].tolist(), args.batch_size)
    df["response_tokens"] = count_tokens(tokenizer, df["response"].tolist(), args.batch_size)

    keep = (df["prompt_tokens"] <= args.max_prompt_tokens) & (
        df["response_tokens"] <= args.max_response_tokens
    )
    print(
        f"[filter] prompt>{args.max_prompt_tokens}: {(df['prompt_tokens'] > args.max_prompt_tokens).sum()}, "
        f"response>{args.max_response_tokens}: {(df['response_tokens'] > args.max_response_tokens).sum()}"
    )
    df = df[keep].reset_index(drop=True)
    print(f"[filter] kept {len(df)} rows")
    if df.empty:
        raise RuntimeError("all samples were filtered out")

    # ------------------------------------------------------------------ step 3
    system_prompt = common_prefix(df["query"].tolist()).strip()
    print(f"[system] common prefix ({len(system_prompt)} chars):\n{system_prompt}\n")

    def build_messages(row):
        query = row["query"]
        if system_prompt and query.startswith(system_prompt):
            query = query[len(system_prompt) :].lstrip()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

    df["prompt"] = df.apply(build_messages, axis=1)

    # ------------------------------------------------------------------ step 4
    df["examples"] = df["query"].map(extract_examples)
    print(f"[examples] rows without example: {(df['examples'].map(len) == 0).sum()}")

    # ------------------------------------------------------------------ step 5
    if "source" in df.columns:
        df = df.rename(columns={"source": "data_source"})
    df["target"] = df["response"].map(lambda c: [{"role": "assistant", "content": c}])

    df = df.drop(columns=[c for c in ["messages", "query", "response"] if c in df.columns])

    cols = [c for c in ["data_source", "prompt", "target", "examples"] if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[save] {len(df)} rows -> {args.output}")
    print(df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
