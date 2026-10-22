"""Dataset interface for the competitive-programming (code) parquet files.

The parquet produced by ``data/prepare_openr1_code.py`` has the columns::

    data_source : str   e.g. "open-r1/codeforces-cots"
    prompt      : list[{"role": "system"|"user", "content": str}]
    target      : list[{"role": "assistant", "content": str}]   (reference CoT + code)
    examples    : list[{"input": str, "output": str}]           (public test cases)

This module provides

* :class:`CodeDataset` / :func:`iter_code_examples` - a light-weight read-only
  interface to walk through the samples (problem statement, test cases, ...);
* :func:`attach_code_reward_model` - the glue used by the RL datasets so that
  ``non_tensor_batch['reward_model']['ground_truth']`` carries the test cases,
  which is exactly what the reward manager expects. Thanks to this, training on
  the code parquet only requires changing ``data.train_files`` / ``data.val_files``.
"""

import json
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import pandas as pd

# reuse the single source of truth for test-case normalisation
from verl.utils.reward_score.code_exec import normalize_test_cases

CODE_REWARD_STYLE = "code_exec"
EXAMPLES_KEY = "examples"

# columns that must not reach the dataloader collate_fn: nested / ragged values
# would be turned into 2-D object arrays and break DataProto batching.
_DROP_KEYS = (EXAMPLES_KEY, "num_tokens", "prompt_tokens", "response_tokens")


def normalize_examples(value: Any) -> List[Dict[str, str]]:
    """Coerce the ``examples`` cell into ``[{"input": ..., "output": ...}, ...]``."""
    return normalize_test_cases(value)


def dumps_examples(value: Any) -> str:
    """Serialize the test cases so they can be stored as a plain string."""
    return json.dumps(normalize_examples(value), ensure_ascii=False)


def is_code_row(row: Dict[str, Any]) -> bool:
    return EXAMPLES_KEY in row and row.get(EXAMPLES_KEY) is not None


def attach_code_reward_model(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Build ``reward_model`` from ``examples`` and drop non-collatable columns.

    * If the row already carries a ``reward_model`` (math-style data) it is kept
      untouched, so this helper is a no-op for the existing datasets.
    * Otherwise the public test cases are JSON-serialized into
      ``reward_model.ground_truth``.
    """
    examples = row_dict.get(EXAMPLES_KEY, None)
    if row_dict.get("reward_model", None) is None and examples is not None:
        row_dict["reward_model"] = {
            "style": CODE_REWARD_STYLE,
            "ground_truth": dumps_examples(examples),
        }
    for key in _DROP_KEYS:
        row_dict.pop(key, None)
    return row_dict


def _messages_to_list(value: Any) -> List[Dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    return [dict(m) for m in list(value)]


def _role_content(messages: Sequence[Dict[str, str]], role: str) -> str:
    for msg in messages:
        if msg.get("role") == role:
            return msg.get("content", "")
    return ""


class CodeDataset:
    """Read-only, on-demand view over one or several code parquet files.

    Example::

        ds = CodeDataset("data/openr1-code.parquet")
        print(len(ds))
        item = ds[0]
        item["question"]      # problem statement (user turn)
        item["system"]        # shared system prompt
        item["examples"]      # [{"input": ..., "output": ...}, ...]
        item["target"]        # reference solution (CoT + ```cpp block)

        for item in ds:       # lazy iteration, one row materialised at a time
            ...
    """

    def __init__(
        self,
        parquet_files: Union[str, Sequence[str]],
        prompt_key: str = "prompt",
        target_key: str = "target",
        examples_key: str = EXAMPLES_KEY,
        columns: Optional[Sequence[str]] = None,
    ):
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        self.parquet_files = list(parquet_files)
        self.prompt_key = prompt_key
        self.target_key = target_key
        self.examples_key = examples_key
        self.dataframe: pd.DataFrame = pd.concat(
            [pd.read_parquet(f, columns=columns) for f in self.parquet_files],
            ignore_index=True,
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    @property
    def columns(self) -> List[str]:
        return self.dataframe.columns.tolist()

    def __getitem__(self, item: int) -> Dict[str, Any]:
        row = self.dataframe.iloc[item].to_dict()
        messages = _messages_to_list(row.get(self.prompt_key))
        target = _messages_to_list(row.get(self.target_key))
        return {
            "index": int(item),
            "data_source": row.get("data_source", ""),
            "system": _role_content(messages, "system"),
            "question": _role_content(messages, "user"),
            "prompt": messages,
            "target": target[0]["content"] if target else "",
            "examples": normalize_examples(row.get(self.examples_key)),
            "num_tokens": row.get("num_tokens"),
            "prompt_tokens": row.get("prompt_tokens"),
            "response_tokens": row.get("response_tokens"),
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]


def iter_code_examples(
    parquet_files: Union[str, Sequence[str]],
    limit: Optional[int] = None,
    **kwargs: Any,
) -> Iterator[Dict[str, Any]]:
    """Convenience generator over :class:`CodeDataset`."""
    dataset = CodeDataset(parquet_files, **kwargs)
    for i, item in enumerate(dataset):
        if limit is not None and i >= limit:
            return
        yield item


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", nargs="?", default="data/openr1-code.parquet")
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    ds = CodeDataset(args.parquet)
    print(f"rows={len(ds)} columns={ds.columns}")
    for item in iter_code_examples(args.parquet, limit=args.limit):
        print("=" * 80)
        print("data_source:", item["data_source"])
        print("system     :", item["system"])
        print("question   :", item["question"][:300], "...")
        print("examples   :", item["examples"])
        print("target     :", item["target"][:200], "...")
