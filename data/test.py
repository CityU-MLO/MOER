import os
import re
from pathlib import Path

SPLIT_KEYWORDS = [
    r"\nlet me ", r"\nlet's ", r"\nfinally,", r"\nfirst,", r"\nnext,", r"\nstep "
    r"\nnote that ", r"\nnow,", r"\nwait,", r"\nso,", r"\nthen,", r"\ntherefore,"
]
pattern = r'(?i)(' + '|'.join(SPLIT_KEYWORDS) + r')'


def main():
    import pandas as pd

    path = Path(os.environ.get("LUFFY_DATA_DIR", str(Path(__file__).resolve().parent)))
    train_df = pd.read_parquet(path / 'openr1.parquet')
    for i in range(len(train_df)):
        if train_df.iloc[i]['target'] is not None:
            text = train_df.iloc[i]['target'][0]['content'][len('<think>\n'):]
            text = text.split('</think>\n')
            assert len(text) == 2
            think = text[0]
            solution = text[-1]

            # Split the thoughts into segments starting with separator keywords.
            parts = re.split(pattern, think, flags=re.IGNORECASE)
            # parts = [text0, kw1, text1, kw2, text2, ...]
            # Keep the text preceding the first keyword as the first segment.
            think_steps = [parts[0]] + [parts[i] + parts[i+1] for i in range(1, len(parts)-1, 2)]
            num_steps = len(think_steps)
            print(f"Number of thinking steps: {num_steps}\n")
            if i == 14:
                for j, step in enumerate(think_steps):
                    print(f"--- Step {j+1} ---\n{step}\n")

            if i == 20:
                break


if __name__ == "__main__":
    main()
