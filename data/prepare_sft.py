"""Convert local expert traces to OpenRLHF SFT JSONL without mutating input rows."""
import argparse
import copy
import json
import os
from pathlib import Path


def as_messages(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    return copy.deepcopy(value)


def main():
    data_dir = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=data_dir / "openr1.parquet")
    parser.add_argument("--output", type=Path, default=data_dir / "openrlhf_sft/train.jsonl")
    args = parser.parse_args()
    import pandas as pd

    frame = pd.read_parquet(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for row in frame.to_dict("records"):
            target = as_messages(row["target"])
            if not target:
                continue
            content = target[0]["content"]
            if content.startswith("<think>\n"):
                target[0]["content"] = content[len("<think>\n"):]
            item = {"prompt": as_messages(row["prompt"]), "target": target}
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    print(f"Prepared {count} SFT rows. Keep generated JSONL private unless reviewed.")


if __name__ == "__main__":
    main()
