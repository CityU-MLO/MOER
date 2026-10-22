import os
from pathlib import Path


def main():
    import pandas as pd

    data_dir = Path(os.environ.get("LUFFY_DATA_DIR", str(Path(__file__).resolve().parent)))
    output_path = data_dir / "valid_aime25_256.parquet"
    df = pd.read_parquet(data_dir / "valid.all.parquet")
    df = df[df["data_source"] == "aime25"]

    # Append seven copies immediately after each original sample.
    rows = []
    for _, row in df.iterrows():
        rows.append(row)
        for _ in range(7):
            rows.append(row.copy())
    df = pd.DataFrame(rows).reset_index(drop=True)

    df.to_parquet(output_path, index=False)

    df = pd.read_parquet(output_path)
    df = df[df["data_source"] == "aime25"]
    print(df["prompt"].iloc[0])
    print(df["prompt"].iloc[256])


if __name__ == "__main__":
    main()
