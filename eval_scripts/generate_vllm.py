import os
import json
import pandas as pd
import numpy as np

from eval_utils import (
    dataset_answers, decoded_item, ensure_output_directory, input_key,
    read_decoded_cache, to_jsonable, validate_generation,
)

# Generation and math grading dependencies are loaded only when needed.
THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


def timeout(timeout_seconds: int = 10):
    def decorator(func):
        if os.name != 'posix':
            return func
        import signal

        def handler(signum, frame):
            raise TimeoutError('verify timed out!')

        def wrapper(*args, **kwargs):
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


@timeout(timeout_seconds=10)
def labeling_responses(responses: list[str], golden_answer: str):
    from math_verify import parse, verify

    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ['$' + str(golden_answer) + '$'] * len(responses)))
    return list(map(verify, golden_answers, predict_answers))


_CODE_EXEC = None


def get_code_exec():
    """Load the existing local code reward, falling back to installed verl.

    This executes untrusted programs on the host, NOT in a security sandbox.
    """
    global _CODE_EXEC
    if _CODE_EXEC is None:
        local_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..',
            'luffy', 'verl', 'verl', 'utils', 'reward_score', 'code_exec.py',
        ))
        if os.path.exists(local_path):
            import importlib.util
            import sys
            spec = importlib.util.spec_from_file_location('eval_scripts_code_exec', local_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules['eval_scripts_code_exec'] = mod
            spec.loader.exec_module(mod)
            _CODE_EXEC = mod
        else:
            from verl.utils.reward_score import code_exec as _mod
            _CODE_EXEC = _mod
    return _CODE_EXEC


def prepare_code_text(generated_text, prompt_for_think, add_think_before_answer=False):
    """Restore the training-time think prefix without inserting it twice."""
    if THOUGHT_DELIMITER_START not in generated_text:
        if prompt_for_think.endswith(THOUGHT_DELIMITER_START + '\n') or add_think_before_answer is True:
            generated_text = THOUGHT_DELIMITER_START + '\n' + generated_text
    return generated_text


def make_conv_zero(question):
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content


def make_conv_zero_code(question):
    question = question + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content


def make_conv_prime_sft(question, tokenizer):
    content = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    msg = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def simplerl_template(question: str):
    return (
        '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        + question
        + '\nPlease reason step by step, and put your final answer within\\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n'
    )


def main(input_file, output_file, model_path, debug=False, remove_system=True, template='own', temperature=0.6, top_p=1.0, max_tokens=8192, n=1, force_generate=True, add_think_before_answer=False, add_oat_evaluate=False, any_true=False, skip_scoring=False, output_eval=None, no_split_think=False, code_num_workers=16, print_sample=False, include_prompt=False):
    """Evaluate samples; raw prompt logging/storage requires explicit opt-in.

    print_sample=True logs example prompts. include_prompt=True stores original
    prompts in both decoded and scored JSONL files. Generated text and answers
    remain in output files and may themselves contain sensitive information.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    df = pd.read_parquet(input_file)
    answer_mode, source_answers = dataset_answers(df)
    messages = [to_jsonable(message) for message in df['prompt'].tolist()]
    if remove_system:
        messages = [message[1:] if message and message[0].get('role') == 'system' else message
                    for message in messages]
    if any(not message for message in messages):
        raise ValueError('Each input row must contain a nonempty conversation')
    keys = [input_key(message, answer) for message, answer in zip(messages, source_answers)]
    data_sources = [source for source in df['data_source'].tolist() for _ in range(n)]
    expected_keys = [key for key in keys for _ in range(n)]
    ensure_output_directory(output_file)
    dec_output_path = output_file.replace('.jsonl', '') + '.decoded.jsonl'

    items = None
    if not force_generate and os.path.exists(dec_output_path):
        items = read_decoded_cache(dec_output_path)
        if (items is None or len(items) != len(expected_keys)
                or any(item.get('input_key') != key or not isinstance(item.get('generated_text'), str)
                       or not isinstance(item.get('prompt_for_think'), str)
                       or (include_prompt and not isinstance(item.get('prompt'), str))
                       for item, key in zip(items, expected_keys))):
            print('Decoded cache does not match the current input/sample count; regenerating.')
            items = None
        else:
            print('Found matching decoded file, skip decoding...')
    if items is None:
        if print_sample:
            print('Example messages:', messages[0])
        print(f'temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}, n: {n}')
        outputs = generate_vllm(messages, model_path, template=template, temperature=temperature,
                                top_p=top_p, max_tokens=max_tokens, n=n, print_sample=print_sample)
        validate_generation(outputs, len(messages), n)
        items = []
        for i, output in enumerate(outputs):
            for completion in output.outputs:
                item = decoded_item(output.prompt, source_answers[i], keys[i], include_prompt)
                item['generated_text'] = completion.text
                items.append(item)
        with open(dec_output_path, 'w', encoding='utf-8') as file:
            for item in items:
                file.write(json.dumps(item) + '\n')

    outputs = [item['generated_text'] for item in items]
    prompts = [item.get('prompt', item['prompt_for_think']) for item in items]
    answers = [answer for answer in source_answers for _ in range(n)]
    if skip_scoring:
        return

    from collections import defaultdict
    from tqdm import tqdm
    rets = defaultdict(list)
    save_data = []
    avg = 0
    print('Scoring...')
    code_scores = None
    code_exec_score = None
    if answer_mode == 'code':
        code_exec_score = get_code_exec().compute_score

        def _score_idx(idx):
            text = prepare_code_text(outputs[idx], prompts[idx], add_think_before_answer)
            try:
                return idx, float(code_exec_score(text, answers[idx])) > 0
            except Exception:
                return idx, False

        if code_num_workers and code_num_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            print(f'parallel code scoring with {code_num_workers} workers')
            with ThreadPoolExecutor(max_workers=code_num_workers) as pool:
                code_scores = dict(tqdm(pool.map(_score_idx, range(len(outputs))), total=len(outputs)))

    diff_cnt = 0
    for i in tqdm(range(len(outputs)), total=len(outputs)):
        generated_text = outputs[i]
        prompt = prompts[i]
        answer = answers[i]
        if answer_mode == 'code':
            generated_text = prepare_code_text(generated_text, prompt, add_think_before_answer)
            labels = [code_scores[i] if code_scores is not None else _score_idx(i)[1]]
        else:
            labels = None
            think_format = (prompt.endswith(THOUGHT_DELIMITER_START + '\n')
                            or add_think_before_answer is True)
            if think_format:
                generated_text = prepare_code_text(generated_text, prompt, add_think_before_answer)
            if think_format and not no_split_think:
                if THOUGHT_DELIMITER_END in generated_text:
                    generated_text = generated_text.split(THOUGHT_DELIMITER_END, 1)[1]
                else:
                    labels = [False]
            if labels is None:
                try:
                    labels = labeling_responses([generated_text], answer)
                except ImportError:
                    raise
                except Exception:
                    labels = [False]
            if add_oat_evaluate:
                from oat_math_grader import boxed_reward_fn as oat_evaluate
                new_label = oat_evaluate(generated_text, answer, fast=False)[1] == 1.0
                if any_true is True:
                    if not labels[0] and new_label:
                        diff_cnt += 1
                    labels = [bool(labels[0]) or new_label]
                else:
                    labels = [new_label]
        correctness = bool(labels[0])
        rets[data_sources[i]].append(correctness)
        item = {'generated_text': generated_text, 'answer': answer, 'correctness': correctness}
        if include_prompt:
            item['prompt'] = prompt
        save_data.append(item)
        avg += int(correctness)

    print('accuracy: ', avg / len(outputs))
    print('diff_cnt: ', diff_cnt)
    accs = []
    for data_source, labels in rets.items():
        acc = float(np.array(labels).mean())
        print(f'{data_source}: {acc}')
        accs.append(acc)
    # Preserve the original unweighted mean across data sources.
    print('avg acc: ', float(np.array(accs).mean()))
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in save_data:
            file.write(json.dumps(item) + '\n')


def generate_vllm(messages, model_path, template='own', temperature=0.6, top_p=0.95, max_tokens=8192, n=1, print_sample=False):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n)
    print(torch.cuda.device_count())
    llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.85)

    gen_prompts = []
    for i, cur_message in enumerate(messages):
        if template == 'own':
            gen_prompt = tokenizer.apply_chat_template(cur_message, tokenize=False, add_generation_prompt=True)
        elif template == 'simplerl':
            gen_prompt = simplerl_template(cur_message[0]['content'])
        elif template == 'qwen':
            gen_prompt = apply_qwen_math_template(cur_message[0]['content'])
        elif template == 'prime':
            gen_prompt = make_conv_zero(cur_message[0]['content'])
        elif template == 'prime_sft':
            gen_prompt = make_conv_prime_sft(cur_message[0]['content'], tokenizer)
        elif template == 'prime_code':
            gen_prompt = make_conv_zero_code(cur_message[0]['content'])
        elif template == 'no':
            gen_prompt = cur_message[0]['content']
        else:
            raise ValueError(f'Invalid template: {template}')
        gen_prompts.append(gen_prompt)
        if i == 0 and print_sample:
            print('Example input:', gen_prompt)
    return llm.generate(gen_prompts, sampling_params)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
