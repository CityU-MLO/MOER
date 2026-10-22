import os
import json
import numpy as np
import pandas as pd

from generate_vllm import (
    THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, labeling_responses,
    generate_vllm, get_code_exec, prepare_code_text,
)
from eval_utils import (
    dataset_answers, decoded_item, ensure_output_directory, input_key,
    read_decoded_cache, to_jsonable as _to_jsonable, validate_generation,
)


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: 1 - C(n - c, k) / C(n, k).

    n is the sample count, c the correct count, and k the requested pass@k.
    """
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError('pass@k requires 0 <= c <= n and 1 <= k <= n')
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _prompt_key(prompt) -> str:
    return json.dumps(_to_jsonable(prompt), sort_keys=True, ensure_ascii=False)


def _group_consecutive(prompts, answers):
    """Group consecutive identical prompts and require equal sample counts."""
    if not prompts or len(prompts) != len(answers):
        raise ValueError('Prompts and answers must be nonempty and aligned')
    groups = []
    for prompt, answer in zip(prompts, answers):
        key = _prompt_key(prompt)
        if groups and groups[-1]['key'] == key:
            if _prompt_key(answer) != _prompt_key(groups[-1]['answer']):
                raise ValueError('Repeated prompts must have identical answers/test cases')
            groups[-1]['count'] += 1
        else:
            groups.append({'key': key, 'prompt': prompt, 'answer': answer, 'count': 1})
    n = groups[0]['count']
    if not all(group['count'] == n for group in groups):
        raise ValueError('Each problem must have the same number of replicated samples')
    return groups, n


def _score_text(generated_text, prompt_for_think, answer,
                add_think_before_answer=False, no_split_think=False, answer_mode='math'):
    if answer_mode == 'code':
        # Keep full think text for the existing code reward format checks.
        text = prepare_code_text(generated_text, prompt_for_think, add_think_before_answer)
        try:
            return float(get_code_exec().compute_score(text, answer)) > 0, text
        except Exception:
            return False, text

    think_format = (prompt_for_think.endswith(THOUGHT_DELIMITER_START + '\n')
                    or add_think_before_answer is True)
    if think_format:
        generated_text = prepare_code_text(generated_text, prompt_for_think, add_think_before_answer)
    if think_format and not no_split_think:
        if THOUGHT_DELIMITER_END not in generated_text:
            return False, generated_text
        generated_text = generated_text.split(THOUGHT_DELIMITER_END, 1)[1]
    try:
        return bool(labeling_responses([generated_text], answer)[0]), generated_text
    except ImportError:
        raise
    except Exception:
        return False, generated_text


def _cache_matches(problems, groups, n, include_prompt=False):
    """Never score a cache with different problem or per-problem sample counts."""
    return problems is not None and len(problems) == len(groups) and all(
        item.get('input_key') == input_key(group['prompt'], group['answer'])
        and isinstance(item.get('generated_texts'), list)
        and len(item['generated_texts']) == n
        and all(isinstance(text, str) for text in item['generated_texts'])
        and isinstance(item.get('prompt_for_think'), str)
        and (not include_prompt or isinstance(item.get('prompt'), str))
        for item, group in zip(problems, groups)
    )


def main(input_file, output_file, model_path, template='own',
         temperature=0.6, top_p=1.0, max_tokens=8192,
         remove_system=True, force_generate=True,
         add_think_before_answer=False, no_split_think=False,
         code_num_workers=16, print_sample=False, include_prompt=False):
    """Evaluate pass@k using the original consecutive-replica grouping.

    print_sample=True logs example prompts. include_prompt=True stores original
    prompts in decoded and scored JSONL files. Answers and generated text may
    contain sensitive information even when prompt storage is disabled.
    """
    df = pd.read_parquet(input_file)
    answer_mode, answers = dataset_answers(df)
    prompts = [_to_jsonable(prompt) for prompt in df['prompt'].tolist()]
    if remove_system:
        prompts = [prompt[1:] if prompt and prompt[0].get('role') == 'system' else prompt
                   for prompt in prompts]
    if any(not prompt for prompt in prompts):
        raise ValueError('Each input row must contain a nonempty conversation')
    groups, n = _group_consecutive(prompts, answers)
    print(f'{len(groups)} problems, {n} samples per problem')
    ensure_output_directory(output_file)
    dec_output_path = output_file.replace('.jsonl', '') + '.decoded.jsonl'

    problems = None
    if not force_generate and os.path.exists(dec_output_path):
        problems = read_decoded_cache(dec_output_path)
        if not _cache_matches(problems, groups, n, include_prompt):
            print('Decoded cache does not match the current problems/sample counts; regenerating.')
            problems = None
        else:
            print('Found matching decoded file, skip decoding...')
    if problems is None:
        uniq_prompts = [group['prompt'] for group in groups]
        print(f'temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}, n: {n}')
        outputs = generate_vllm(uniq_prompts, model_path, template=template,
                                temperature=temperature, top_p=top_p,
                                max_tokens=max_tokens, n=n, print_sample=print_sample)
        validate_generation(outputs, len(groups), n)
        problems = []
        for output, group in zip(outputs, groups):
            item = decoded_item(output.prompt, group['answer'],
                                input_key(group['prompt'], group['answer']), include_prompt)
            item['generated_texts'] = [completion.text for completion in output.outputs]
            problems.append(item)
        with open(dec_output_path, 'w', encoding='utf-8') as file:
            for item in problems:
                file.write(json.dumps(item) + '\n')

    from tqdm import tqdm
    ks = []
    k = 1
    while k <= n:
        ks.append(k)
        k *= 2
    pass_at_k = {k: [] for k in ks}
    save_data = []
    print('Scoring...')

    code_correctness = None
    if answer_mode == 'code' and code_num_workers and code_num_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        code_exec_score = get_code_exec().compute_score

        def _score_one(job):
            pi, si = job
            item = problems[pi]
            prompt = item.get('prompt', item['prompt_for_think'])
            text = prepare_code_text(item['generated_texts'][si], prompt, add_think_before_answer)
            try:
                return pi, si, float(code_exec_score(text, groups[pi]['answer'])) > 0
            except Exception:
                return pi, si, False

        jobs = [(pi, si) for pi in range(len(problems)) for si in range(n)]
        print(f'parallel code scoring with {code_num_workers} workers')
        code_correctness = {}
        with ThreadPoolExecutor(max_workers=code_num_workers) as pool:
            for pi, si, ok in tqdm(pool.map(_score_one, jobs), total=len(jobs)):
                code_correctness[(pi, si)] = ok

    for pi, item in enumerate(tqdm(problems, total=len(problems))):
        prompt = item.get('prompt', item['prompt_for_think'])
        answer = groups[pi]['answer']
        texts = item['generated_texts']
        if code_correctness is not None:
            correctness = [code_correctness[(pi, si)] for si in range(n)]
        else:
            correctness = [_score_text(text, prompt, answer,
                                       add_think_before_answer=add_think_before_answer,
                                       no_split_think=no_split_think,
                                       answer_mode=answer_mode)[0] for text in texts]
        c = int(sum(correctness))
        for k in ks:
            pass_at_k[k].append(estimate_pass_at_k(n, c, k))
        result = {'answer': answer, 'correctness': correctness,
                  'num_correct': c, 'num_samples': n}
        if include_prompt:
            result['prompt'] = prompt
        save_data.append(result)

    print('\n===== pass@k =====')
    for k in ks:
        print(f'pass@{k}: {float(np.mean(pass_at_k[k])):.4f}')
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in save_data:
            file.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
