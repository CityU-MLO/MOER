"""Shared schema, serialization, and decoded-cache helpers."""

import hashlib
import json
import math
import os

import numpy as np

TEST_KEYS = ('examples', 'test_cases', 'tests', 'cases')
CODE_STYLES = {'code', 'coding', 'code_exec', 'prime_code'}
MATH_STYLES = {'math', 'rule'}


def to_jsonable(obj):
    """Convert nested numpy values to native JSON-compatible values."""
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _present(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _has_tests(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    if isinstance(value, dict):
        return (any(key in value for key in TEST_KEYS) or 'input' in value or 'output' in value
                or ('inputs' in value and 'outputs' in value))
    return isinstance(value, list) and bool(value) and all(
        (isinstance(case, dict) and ('input' in case or 'output' in case))
        or (isinstance(case, list) and len(case) == 2) for case in value
    )


def dataset_answers(df):
    """Identify test-case schemas before considering math reward metadata."""
    modes, answers = [], []
    for row in df.to_dict(orient='records'):
        row = to_jsonable(row)
        reward = row.get('reward_model')
        reward = reward if isinstance(reward, dict) else {}
        answer = reward.get('ground_truth')
        tests = next((row[key] for key in TEST_KEYS if _present(row.get(key))), None)
        if tests is None:
            tests = next((reward[key] for key in TEST_KEYS if _present(reward.get(key))), None)
        style = reward.get('style', row.get('style', 'rule'))
        if tests is not None or _has_tests(answer) or style in CODE_STYLES:
            modes.append('code')
            answers.append(tests if tests is not None else answer)
        elif style in MATH_STYLES and 'ground_truth' in reward:
            modes.append('math')
            answers.append(answer)
        else:
            raise ValueError('Input rows must contain code tests or a supported math reward_model schema')
    if not modes:
        raise ValueError('Input parquet is empty')
    if len(set(modes)) != 1:
        raise ValueError('Code and math rows must be evaluated separately')
    if modes[0] == 'code':
        print('Code scoring executes untrusted programs locally; this is NOT a sandbox.')
    return modes[0], answers


def input_key(prompt, answer):
    """Identify cached input rows without storing the original prompt."""
    serialized = json.dumps(to_jsonable([prompt, answer]), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def ensure_output_directory(output_file):
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)


def read_decoded_cache(path):
    """Return a parsed cache, or None when it needs regeneration."""
    try:
        with open(path, encoding='utf-8') as file:
            items = [json.loads(line) for line in file]
        if all(isinstance(item, dict) for item in items):
            return items
    except (OSError, ValueError):
        pass
    return None


def validate_generation(outputs, num_prompts, n):
    if len(outputs) != num_prompts or any(len(output.outputs) != n for output in outputs):
        raise ValueError('Generation output count does not match the input and requested samples')


def decoded_item(prompt, answer, key, include_prompt=False):
    """Retain only the prompt suffix required by the existing scoring rules."""
    item = {'answer': to_jsonable(answer), 'input_key': key,
            'prompt_for_think': '<think>\n' if prompt.endswith('<think>\n') else ''}
    if include_prompt:
        item['prompt'] = prompt
    return item
