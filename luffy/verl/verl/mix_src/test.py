import argparse
import os


def main(argv=None):
    parser = argparse.ArgumentParser(description='Inspect the chat template using a local tokenizer')
    parser.add_argument(
        '--tokenizer-path',
        default=os.environ.get('MIX_TOKENIZER_PATH', 'models/Qwen2.5-1.5B-Instruct'),
        help='Local tokenizer directory; can also be set via MIX_TOKENIZER_PATH',
    )
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.expanduser(args.tokenizer_path), local_files_only=True
    )
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello, how are you?'},
    ]
    print(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))


if __name__ == '__main__':
    main()
