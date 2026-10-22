import argparse
import os


messages = [
    {
        'role': 'system',
        'content': 'You are a helpful assistant.'
    },
    {
        'role': 'user',
        'content': 'Hello, how are you?'
    }
]


def main():
    tokenizer_path = os.environ.get("LUFFY_TOKENIZER_PATH")
    parser = argparse.ArgumentParser(description="Inspect the local tokenizer chat template")
    parser.add_argument("--tokenizer", default=tokenizer_path, required=not tokenizer_path,
                        help="Local tokenizer directory (or set LUFFY_TOKENIZER_PATH)")
    args = parser.parse_args()
    if not os.path.isdir(args.tokenizer):
        raise FileNotFoundError(f"Local tokenizer directory not found: {args.tokenizer}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))


if __name__ == "__main__":
    main()
