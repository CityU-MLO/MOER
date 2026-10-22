"""Prepare held-out RL data for the SFT-then-RL baseline."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Elliott/Openr1-Math-48k-Complement")
    parser.add_argument("--output", type=Path,
                        default=Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent)) / "openr1.complement.parquet")
    args = parser.parse_args()
    from datasets import load_dataset
    import pandas as pd

    dataset = load_dataset(args.dataset, split="train")
    frame = pd.DataFrame(list(dataset))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"Prepared {len(frame)} held-out RL rows.")


if __name__ == "__main__":
    main()
