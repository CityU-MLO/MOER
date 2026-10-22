"""Filter out samples where no C++ code block can be extracted from the reference solution."""
import sys, importlib.util, os
import pandas as pd
import time

# Load code_exec module avoiding math.py shadow
spec = importlib.util.spec_from_file_location(
    'code_exec',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../luffy/verl/verl/utils/reward_score/code_exec.py')
)
code_exec = importlib.util.module_from_spec(spec)
_old = sys.path[:]
sys.path = [p for p in sys.path if 'reward_score' not in p]
spec.loader.exec_module(code_exec)
sys.path = _old


def main():
    parquet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openr1-code.parquet')
    df = pd.read_parquet(parquet_path)
    total = len(df)
    print(f"Total samples: {total}", flush=True)

    no_code_indices = []
    for i in range(total):
        row = df.iloc[i]
        target = row['target']
        if target is None or len(target) == 0:
            no_code_indices.append(i)
            continue
        content = target[0]['content'] if isinstance(target[0], dict) else str(target[0])
        code = code_exec.extract_cpp_code(content)
        if code is None:
            no_code_indices.append(i)

    n_removed = len(no_code_indices)
    print(f"Samples without extractable code: {n_removed}", flush=True)
    print(f"Samples with valid code: {total - n_removed}", flush=True)

    # Filter
    df_filtered = df.drop(index=no_code_indices).reset_index(drop=True)
    df_filtered.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df_filtered)} samples to {parquet_path} (removed {n_removed})", flush=True)


if __name__ == "__main__":
    main()
