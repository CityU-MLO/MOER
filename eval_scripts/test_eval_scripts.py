"""Offline regression tests: no GPU, network, training, or generated-code execution."""

import ast
import contextlib
import importlib
import importlib.abc
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent


class FakeArray(list):
    def tolist(self):
        return list(self)

    def mean(self):
        return sum(self) / len(self)

    def __rtruediv__(self, other):
        return FakeArray(other / value for value in self)

    def __rsub__(self, other):
        return FakeArray(other - value for value in self)


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeFrame:
    def __init__(self, rows):
        self.rows = rows
        self.columns = set().union(*(row.keys() for row in rows))

    def __getitem__(self, key):
        return FakeArray(row.get(key) for row in self.rows)

    def to_dict(self, orient):
        assert orient == 'records'
        return self.rows


class BlockHeavyImports(importlib.abc.MetaPathFinder):
    blocked = {'torch', 'vllm', 'transformers', 'math_verify', 'sympy',
               'latex2sympy2_extended', 'pylatexenc', 'oat_math_grader'}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.blocked:
            raise ImportError('Heavy dependency is blocked by the offline test')
        return None


def fake_modules():
    numpy = types.ModuleType('numpy')
    numpy.ndarray = FakeArray
    numpy.generic = FakeScalar
    numpy.array = FakeArray
    numpy.arange = lambda start, stop: FakeArray(range(start, stop))
    numpy.prod = math.prod
    numpy.mean = lambda values: sum(values) / len(values)
    pandas = types.ModuleType('pandas')
    pandas.read_parquet = Mock()
    tqdm = types.ModuleType('tqdm')
    tqdm.tqdm = lambda iterable, **kwargs: iterable
    return {'numpy': numpy, 'pandas': pandas, 'tqdm': tqdm}


def row(name='private-input', source='source-a', code=False):
    item = {'prompt': [{'role': 'user', 'content': name}], 'data_source': source,
            'reward_model': {'style': 'rule', 'ground_truth': '7'}}
    if code:
        item['examples'] = [{'input': '1', 'output': '2'}]
    return item


def generated(messages, model_path, n=1, **kwargs):
    return [types.SimpleNamespace(
        prompt=message[-1]['content'] + '<think>\n',
        outputs=[types.SimpleNamespace(text='reason</think>7') for _ in range(n)],
    ) for message in messages]


class PythonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT, prefix='.eval_python_mock_')
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'new output' / 'result.jsonl'
        self.modules = fake_modules()
        module_patch = patch.dict(sys.modules, self.modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        for name in ('eval_utils', 'generate_vllm', 'pass_at_k_vllm', 'collect_results'):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(ROOT))
        self.addCleanup(sys.path.remove, str(ROOT))
        blocker = BlockHeavyImports()
        sys.meta_path.insert(0, blocker)
        self.addCleanup(sys.meta_path.remove, blocker)
        self.utils = importlib.import_module('eval_utils')
        self.gen = importlib.import_module('generate_vllm')
        self.passk = importlib.import_module('pass_at_k_vllm')
        self.collect = importlib.import_module('collect_results')
        self.modules['pandas'].read_parquet.return_value = FakeFrame([row()])
        self.calls = Mock(side_effect=generated)
        self.gen.generate_vllm = self.calls
        self.passk.generate_vllm = self.calls
        self.gen.labeling_responses = lambda responses, answer: [text.strip() == str(answer) for text in responses]
        self.passk.labeling_responses = self.gen.labeling_responses
        self.code_calls = Mock(return_value=1.0)
        self.code_module = types.SimpleNamespace(compute_score=self.code_calls)
        self.gen._CODE_EXEC = self.code_module
        self.logs = io.StringIO()

    def run_main(self, module=None, **kwargs):
        with contextlib.redirect_stdout(self.logs):
            (module or self.gen).main('mock.parquet', str(self.output), 'mock-model', **kwargs)

    def read_output(self, decoded=False):
        path = Path(str(self.output).replace('.jsonl', '') + '.decoded.jsonl') if decoded else self.output
        return [json.loads(line) for line in path.read_text().splitlines()]

    def set_rows(self, rows):
        self.modules['pandas'].read_parquet.return_value = FakeFrame(rows)

    def test_ast_and_english_source(self):
        for path in ROOT.glob('*'):
            if path.suffix not in {'.py', '.sh'}:
                continue
            source = path.read_text()
            self.assertFalse(any('\u4e00' <= char <= '\u9fff' for char in source), path.name)
            self.assertNotIn('/' + 'apdcephfs/', source)
            if path.suffix == '.py':
                ast.parse(source, filename=path.name)
        tree = ast.parse((ROOT / 'generate_vllm.py').read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        defaults = dict(zip([arg.arg for arg in main.args.args][-len(main.args.defaults):], main.args.defaults))
        self.assertEqual(defaults['n'].value, 1)
        self.assertEqual(defaults['max_tokens'].value, 8192)

    def test_code_schema_has_priority_over_math_metadata(self):
        with contextlib.redirect_stdout(self.logs):
            for key in self.utils.TEST_KEYS:
                item = row()
                item[key] = [{'input': '1', 'output': '2'}]
                mode, answers = self.utils.dataset_answers(FakeFrame([item]))
                self.assertEqual(mode, 'code')
                self.assertEqual(answers[0], item[key])

    def test_reward_styles_and_nested_test_schemas(self):
        schemas = [
            {'style': style, 'ground_truth': '[{"input":"1","output":"2"}]'}
            for style in self.utils.CODE_STYLES
        ] + [
            {'style': 'rule', 'ground_truth': value} for value in (
                {'tests': [{'input': '1', 'output': '2'}]},
                {'inputs': ['1'], 'outputs': ['2']},
                [['1', '2']],
                '[{"input":"1","output":"2"}]',
            )
        ] + [{'style': 'rule', 'ground_truth': 'math answer', 'tests': []}]
        with contextlib.redirect_stdout(self.logs):
            for reward in schemas:
                item = row()
                item['reward_model'] = reward
                self.assertEqual(self.utils.dataset_answers(FakeFrame([item]))[0], 'code')
            for style in ('rule', 'math'):
                item = row()
                item['reward_model']['style'] = style
                item['examples'] = None
                self.assertEqual(self.utils.dataset_answers(FakeFrame([item]))[0], 'math')

    def test_unknown_and_mixed_schemas_fail(self):
        unknown = row()
        unknown['reward_model']['style'] = 'model'
        for rows in ([unknown], [row(), row('code', code=True)], []):
            with self.assertRaises(ValueError):
                self.utils.dataset_answers(FakeFrame(rows))

    def test_nested_numpy_conversion(self):
        value = {'array': FakeArray([FakeScalar(3), FakeArray([FakeScalar(4)])])}
        self.assertEqual(self.utils.to_jsonable(value), {'array': [3, [4]]})

    def test_multi_sample_source_alignment_and_privacy(self):
        self.set_rows([row('secret-one', 'source-a'), row('secret-two', 'source-b')])
        self.gen.labeling_responses = Mock(side_effect=[[True], [False], [True], [True]])
        self.run_main(n=2, add_think_before_answer=True)
        self.assertEqual(len(self.read_output()), 4)
        self.assertIn('source-a: 0.5', self.logs.getvalue())
        self.assertIn('source-b: 1.0', self.logs.getvalue())
        for items in (self.read_output(), self.read_output(decoded=True)):
            self.assertTrue(all('prompt' not in item for item in items))
        self.assertNotIn('secret-one', self.logs.getvalue())
        self.assertNotIn('secret-one', self.output.read_text())
        self.assertTrue(all(len(item['input_key']) == 64 for item in self.read_output(decoded=True)))

    def test_unweighted_source_mean_is_preserved(self):
        self.set_rows([row('p1', 'a'), row('p2', 'a'), row('p3', 'b')])
        self.gen.labeling_responses = Mock(side_effect=[[True], [False], [True]])
        self.run_main()
        self.assertIn('accuracy:  0.6666666666666666', self.logs.getvalue())
        self.assertIn('avg acc:  0.75', self.logs.getvalue())

    def test_explicit_prompt_options(self):
        self.run_main(print_sample=True, include_prompt=True)
        self.assertIn('private-input', self.logs.getvalue())
        for items in (self.read_output(), self.read_output(decoded=True)):
            self.assertIn('private-input', items[0]['prompt'])
        self.assertTrue(self.calls.call_args.kwargs['print_sample'])

    def test_each_row_removes_only_its_own_system_message(self):
        first, second = row('first'), row('second')
        second['prompt'].insert(0, {'role': 'system', 'content': 'private system'})
        self.set_rows([first, second])
        self.run_main(skip_scoring=True)
        messages = self.calls.call_args.args[0]
        self.assertEqual([message[0]['role'] for message in messages], ['user', 'user'])
        self.assertEqual(messages[0][0]['content'], 'first')

    def test_output_alignment_checks_all_requests(self):
        self.set_rows([row('one'), row('two')])
        for counts in ([2], [2, 1], [2, 2, 2]):
            self.calls.side_effect = lambda *args, counts=counts, **kwargs: [
                types.SimpleNamespace(prompt='mock', outputs=[types.SimpleNamespace(text='7') for _ in range(count)])
                for count in counts
            ]
            with self.assertRaises(ValueError):
                self.run_main(n=2)
        self.assertFalse(self.output.exists())

    def test_regular_cache_matches_and_rejects_wrong_n(self):
        self.run_main(n=2)
        self.run_main(n=2, force_generate=False)
        self.assertEqual(self.calls.call_count, 1)
        self.assertEqual(len(self.read_output()), 2)
        self.run_main(n=1, force_generate=False)
        self.assertEqual(self.calls.call_count, 2)
        self.assertEqual(len(self.read_output()), 1)

    def test_regular_cache_rejects_changed_input_and_truncation(self):
        self.run_main(n=2)
        self.set_rows([row('changed')])
        self.run_main(n=2, force_generate=False)
        self.assertEqual(self.calls.call_count, 2)
        cache = Path(str(self.output).replace('.jsonl', '') + '.decoded.jsonl')
        cache.write_text('{invalid JSON}\n')
        self.run_main(n=2, force_generate=False)
        self.assertEqual(self.calls.call_count, 3)

    def test_skip_scoring_does_not_load_code_execution(self):
        self.set_rows([row(code=True)])
        self.gen.get_code_exec = Mock(side_effect=AssertionError('Scoring must be skipped'))
        self.run_main(skip_scoring=True)
        self.assertFalse(self.output.exists())
        self.assertEqual(len(self.read_output(decoded=True)), 1)

    def test_code_scoring_serial_parallel_and_idempotent_prefix(self):
        self.set_rows([row(code=True)])
        for workers in (1, 2):
            self.code_calls.side_effect = [1.0, -1.0, 0.0, RuntimeError('Mock failure')]
            self.run_main(n=4, code_num_workers=workers, add_think_before_answer=True)
            self.assertEqual([item['correctness'] for item in self.read_output()], [True, False, False, False])
        for call in self.code_calls.call_args_list:
            self.assertEqual(call.args[0].count('<think>'), 1)
        self.assertEqual(self.gen.prepare_code_text('<think>reason</think>code', '', True), '<think>reason</think>code')

    def test_math_think_split_uses_first_delimiter(self):
        self.passk.labeling_responses = Mock(return_value=[True])
        ok, text = self.passk._score_text('reason</think>answer</think>tail', '', 'answer', True)
        self.assertTrue(ok)
        self.assertEqual(text, 'answer</think>tail')
        self.assertFalse(self.passk._score_text('missing delimiter', '', '7', True)[0])
        self.passk._score_text('reason</think>7', '', '7', True, True)
        self.assertTrue(self.passk.labeling_responses.call_args.args[0][0].startswith('<think>'))

    def test_pass_estimator_matches_combinatorial_definition(self):
        for n in (1, 2, 4, 8, 256):
            for c in sorted({0, 1, n // 2, n}):
                for k in sorted({1, min(2, n), n}):
                    expected = 1 - (math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 0)
                    self.assertAlmostEqual(self.passk.estimate_pass_at_k(n, c, k), expected)
        for values in ((0, 0, 1), (2, 3, 1), (2, 1, 3)):
            with self.assertRaises(ValueError):
                self.passk.estimate_pass_at_k(*values)

    def test_consecutive_groups_reject_invalid_replication(self):
        for prompts, answers in (([], []), (['a'], []), (['a', 'a', 'b'], ['7'] * 3),
                                 (['a', 'a'], ['7', '8'])):
            with self.assertRaises(ValueError):
                self.passk._group_consecutive(prompts, answers)

    def test_pass_sampling_preserves_256_and_single_sample(self):
        for n in (1, 256):
            self.set_rows([row('p1')] * n + [row('p2')] * n)
            self.run_main(self.passk)
            self.assertEqual(self.calls.call_args.kwargs['n'], n)
            self.assertEqual(len(self.calls.call_args.args[0]), 2)
            self.assertEqual([item['num_samples'] for item in self.read_output()], [n, n])
            self.assertTrue(all('prompt' not in item for item in self.read_output()))
            if n == 256:
                self.assertIn('pass@256: 1.0000', self.logs.getvalue())

    def test_pass_cache_validates_count_samples_and_identity(self):
        self.set_rows([row('p1')] * 2 + [row('p2')] * 2)
        self.run_main(self.passk)
        original = self.read_output(decoded=True)
        self.run_main(self.passk, force_generate=False)
        self.assertEqual(self.calls.call_count, 1)
        bad_caches = [original[:1], original + [original[0]], list(reversed(original))]
        for count in (1, 3):
            cache = json.loads(json.dumps(original))
            cache[1]['generated_texts'] = ['7'] * count
            bad_caches.append(cache)
        for bad in bad_caches:
            path = Path(str(self.output).replace('.jsonl', '') + '.decoded.jsonl')
            path.write_text(''.join(json.dumps(item) + '\n' for item in bad))
            previous = self.calls.call_count
            self.run_main(self.passk, force_generate=False)
            self.assertEqual(self.calls.call_count, previous + 1)
            self.assertEqual([item['num_samples'] for item in self.read_output()], [2, 2])

    def test_pass_code_serial_parallel_and_explicit_prompt(self):
        self.set_rows([row('secret-code', code=True)] * 2)
        for workers in (1, 2):
            self.code_calls.side_effect = [1.0, -1.0]
            self.run_main(self.passk, code_num_workers=workers, include_prompt=True, add_think_before_answer=True)
            self.assertEqual(self.read_output()[0]['correctness'], [True, False])
            self.assertIn('secret-code', self.read_output()[0]['prompt'])
            self.assertIn('pass@1: 0.5000', self.logs.getvalue())
            self.assertIn('pass@2: 1.0000', self.logs.getvalue())

    def test_actual_sampling_params_with_fake_runtime(self):
        captured = {}
        torch = types.ModuleType('torch')
        torch.cuda = types.SimpleNamespace(device_count=lambda: 2)
        transformers = types.ModuleType('transformers')
        tokenizer = types.SimpleNamespace(apply_chat_template=lambda message, **kwargs: message[0]['content'])
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
        vllm = types.ModuleType('vllm')

        class FakeLLM:
            def __init__(self, **kwargs):
                captured['llm'] = kwargs

            def generate(self, prompts, params):
                captured['prompts'] = prompts
                return []

        def sampling_params(**kwargs):
            captured['sampling'] = kwargs
            return kwargs

        vllm.LLM = FakeLLM
        vllm.SamplingParams = sampling_params
        original = importlib.util.module_from_spec(importlib.util.spec_from_file_location('sampling_test', ROOT / 'generate_vllm.py'))
        with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers, 'vllm': vllm}):
            original.__spec__.loader.exec_module(original)
            with contextlib.redirect_stdout(self.logs):
                for template in ('own', 'simplerl', 'qwen', 'prime', 'prime_sft', 'prime_code', 'no'):
                    original.generate_vllm([row()['prompt']], 'mock-model', template=template, max_tokens=123, n=3)
                    self.assertEqual(captured['sampling']['max_tokens'], 123)
                    self.assertEqual(captured['sampling']['n'], 3)
                self.assertNotIn('private-input', self.logs.getvalue())
                original.generate_vllm([row()['prompt']], 'mock-model', max_tokens=456, print_sample=True)
        self.assertEqual(captured['sampling']['max_tokens'], 456)
        self.assertIn('private-input', self.logs.getvalue())

    def test_missing_math_stack_fails_instead_of_scoring_everything_wrong(self):
        self.gen.labeling_responses = Mock(side_effect=ImportError('Mock missing math stack'))
        with self.assertRaises(ImportError):
            self.run_main()
        self.passk.labeling_responses = self.gen.labeling_responses
        with self.assertRaises(ImportError):
            self.passk._score_text('7', '', '7')

    def test_pass_generation_checks_every_problem_sample_count(self):
        self.set_rows([row('p1')] * 2 + [row('p2')] * 2)
        self.calls.side_effect = lambda *args, **kwargs: [
            types.SimpleNamespace(prompt='mock', outputs=[types.SimpleNamespace(text='7')] * count)
            for count in (2, 1)
        ]
        with self.assertRaises(ValueError):
            self.run_main(self.passk)
        self.assertFalse(self.output.exists())

    def test_enabling_prompt_storage_does_not_reuse_promptless_cache(self):
        for module in (self.gen, self.passk):
            self.run_main(module)
            previous = self.calls.call_count
            self.run_main(module, force_generate=False, include_prompt=True)
            self.assertEqual(self.calls.call_count, previous + 1)
            self.assertIn('prompt', self.read_output()[0])

    def test_output_write_errors_propagate(self):
        self.run_main()
        real_open = open

        def checked_open(path, *args, **kwargs):
            if str(path) == str(self.output):
                raise OSError('Mock output write failure')
            return real_open(path, *args, **kwargs)

        with patch('builtins.open', side_effect=checked_open):
            with self.assertRaises(OSError):
                self.run_main(force_generate=False)
        self.run_main(self.passk)
        with patch('builtins.open', side_effect=checked_open):
            with self.assertRaises(OSError):
                self.run_main(self.passk, force_generate=False)

    def test_lazy_math_import_occurs_only_when_labeling(self):
        original = importlib.util.module_from_spec(importlib.util.spec_from_file_location('lazy_test', ROOT / 'generate_vllm.py'))
        original.__spec__.loader.exec_module(original)
        with self.assertRaises(ImportError):
            original.labeling_responses(['7'], '7')

    def test_local_code_exec_integration_loads_without_math_or_execution(self):
        self.gen._CODE_EXEC = None
        with patch('subprocess.run', side_effect=AssertionError('Generated code must not execute')):
            module = self.gen.get_code_exec()
            self.assertTrue(callable(module.compute_score))
            self.assertEqual(module.normalize_test_cases({'tests': [{'input': '1', 'output': '2'}]}),
                             [{'input': '1', 'output': '2'}])

    def test_collector_preserves_full_names_all_metrics_and_scientific_notation(self):
        directory = Path(self.temp.name)
        lines = ['accuracy: 0.5', 'diff_cnt: 12', 'temperature: 0.6, top_p: 1.0']
        lines += [f'bench-{i}/subset: {i / 10}' for i in range(8)]
        lines += ['avg acc: 5e-1', 'pass@1: .25', 'pass@256: 1.0', 'trailing output']
        (directory / 'results.log').write_text('\n'.join(lines))
        (directory / 'empty.log').write_text('no metrics')
        result = self.collect.extract_results_from_files(str(directory))
        metrics = result['results.log']
        self.assertEqual(metrics['avg acc'], 0.5)
        self.assertEqual(metrics['pass@1'], 0.25)
        self.assertEqual(metrics['pass@256'], 1.0)
        self.assertIn('bench-0/subset', metrics)
        self.assertNotIn('diff_cnt', metrics)
        self.assertNotIn('temperature', metrics)
        self.assertNotIn('empty.log', result)


class ShellTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT, prefix='.eval_shell_mock_')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.cwd = self.directory / 'foreign working directory'
        self.cwd.mkdir()
        self.bin = self.directory / 'fake bin'
        self.bin.mkdir()
        self.fake = self.bin / 'python'
        self.fake.write_text(
            '#!' + sys.executable + '\n'
            'import json, os, sys\n'
            'args = sys.argv[1:]\n'
            'keys = ["CUDA_VISIBLE_DEVICES", "LUFFY_CPP_COMPILER", "LUFFY_CODE_COMPILE_TIMEOUT", '
            '"LUFFY_CODE_RUN_TIMEOUT", "LUFFY_CODE_WALL_MULTIPLIER", "LUFFY_CODE_MEMORY_LIMIT_MB", '
            '"LUFFY_CODE_OPT_LEVEL", "LUFFY_CODE_CACHE_SIZE", "LUFFY_REWARD_WORKERS"]\n'
            'record = {"args": args, "cwd": os.getcwd(), "env": {key: os.environ.get(key) for key in keys}}\n'
            'with open(os.environ["MOCK_LOG"], "a") as file: file.write(json.dumps(record) + "\\n")\n'
            'print("MOCK stdout")\n'
            'print("MOCK stderr", file=sys.stderr)\n'
            'sys.exit(23 if os.environ.get("MOCK_FAIL_INPUT") in args else 0)\n'
        )
        self.fake.chmod(0o755)
        self.log = self.directory / 'calls.jsonl'
        self.output = self.directory / 'new results with spaces'
        self.env = {key: value for key, value in os.environ.items()
                    if key not in {'DATA', 'OUTPUT_DIR', 'MODEL_PATH', 'MODEL_NAME', 'TEMPLATE', 'GPU', 'PYTHON'}
                    and not key.startswith('LUFFY_')}
        self.env.update(DATA=str(self.directory / 'data with spaces'), OUTPUT_DIR=str(self.output),
                        MODEL_PATH=str(self.directory / 'model with spaces'), MODEL_NAME='test model',
                        MOCK_LOG=str(self.log), PATH=str(self.bin) + ':/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1')

    def invoke(self, script, **env):
        merged = dict(self.env)
        merged.update(env)
        return subprocess.run(['/bin/bash', str(ROOT / script), '--max_tokens', '321'],
                              cwd=self.cwd, env=merged, capture_output=True, text=True)

    def records(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_batch_benchmarks_cli_and_foreign_cwd(self):
        expected = {
            'eval.sh': ['valid.all.parquet', 'valid.mmlu_pro.parquet', 'valid.arc_c.parquet', 'valid.gpqa.parquet'],
            'pass_at_k.sh': ['valid_aime_256.parquet', 'valid_aime25_256.parquet'],
            'eval_livecodebench.sh': ['livecodebench.parquet', 'valid.arc_c.parquet', 'valid.gpqa.parquet', 'valid.mmlu_pro.parquet'],
        }
        for script, files in expected.items():
            with self.subTest(script=script):
                if self.log.exists():
                    self.log.unlink()
                result = self.invoke(script)
                self.assertEqual(result.returncode, 0, result.stderr)
                records = self.records()
                self.assertEqual(len(records), len(files))
                self.assertEqual([Path(record['args'][record['args'].index('--input_file') + 1]).name
                                  for record in records], files)
                for record in records:
                    args = record['args']
                    self.assertEqual(record['cwd'], str(ROOT.parent))
                    self.assertEqual(args[args.index('--model_path') + 1], self.env['MODEL_PATH'])
                    self.assertEqual(args[args.index('--template') + 1], 'own')
                    self.assertEqual(args[args.index('--max_tokens') + 1], '321')
                    self.assertEqual(record['env']['CUDA_VISIBLE_DEVICES'], '0,1,2,3')
                    self.assertNotIn('--n', args)
                    self.assertEqual(args[args.index('--add_think_before_answer') + 1], 'True')
                    output = Path(args[args.index('--output_file') + 1])
                    self.assertTrue(output.parent.is_dir())
                    log = output.with_suffix('.log').read_text()
                    self.assertIn('MOCK stdout', log)
                    self.assertIn('MOCK stderr', log)
                if script == 'eval_livecodebench.sh':
                    self.assertNotIn('--add_oat_evaluate', records[0]['args'])
                    self.assertTrue(all('--add_oat_evaluate' in record['args'] for record in records[1:]))

    def test_template_selection_and_overrides(self):
        for name, template in [('eurus-2-7b-prime-zero', 'prime'), ('simple-rl-zero', 'qwen')]:
            self.assertEqual(self.invoke('pass_at_k.sh', MODEL_NAME=name).returncode, 0)
            self.assertEqual(self.records()[-1]['args'][self.records()[-1]['args'].index('--template') + 1], template)
        result = self.invoke('eval_livecodebench.sh', GPU='7', TEMPLATE='no', PYTHON=str(self.fake),
                             LUFFY_CPP_COMPILER='custom-c++', LUFFY_CODE_RUN_TIMEOUT='4')
        self.assertEqual(result.returncode, 0, result.stderr)
        record = self.records()[-1]
        self.assertEqual(record['env']['CUDA_VISIBLE_DEVICES'], '7')
        self.assertEqual(record['env']['LUFFY_CPP_COMPILER'], 'custom-c++')
        self.assertEqual(record['env']['LUFFY_CODE_RUN_TIMEOUT'], '4')
        self.assertEqual(record['env']['LUFFY_CODE_COMPILE_TIMEOUT'], '20')
        self.assertEqual(record['env']['LUFFY_CODE_WALL_MULTIPLIER'], '60')
        self.assertEqual(record['env']['LUFFY_CODE_MEMORY_LIMIT_MB'], '1024')
        self.assertEqual(record['env']['LUFFY_CODE_OPT_LEVEL'], '-O2')
        self.assertEqual(record['env']['LUFFY_CODE_CACHE_SIZE'], '50000')
        self.assertEqual(record['env']['LUFFY_REWARD_WORKERS'], '32')

    def test_missing_configuration_fails_without_python_or_value_leak(self):
        for variable in ('DATA', 'OUTPUT_DIR', 'MODEL_PATH'):
            for script in ('eval.sh', 'pass_at_k.sh', 'eval_livecodebench.sh'):
                env = dict(self.env)
                env.pop(variable)
                result = subprocess.run(['/bin/bash', str(ROOT / script)], cwd=self.cwd,
                                        env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Set ' + variable, result.stderr)
                self.assertNotIn(self.env['MODEL_PATH'], result.stderr)
                self.assertEqual(self.records(), [])

    def test_fail_fast_stops_after_first_failed_cli(self):
        for script in ('eval.sh', 'pass_at_k.sh', 'eval_livecodebench.sh'):
            if self.log.exists():
                self.log.unlink()
            first = {'eval.sh': 'valid.all.parquet', 'pass_at_k.sh': 'valid_aime_256.parquet',
                     'eval_livecodebench.sh': 'livecodebench.parquet'}[script]
            result = self.invoke(script, MOCK_FAIL_INPUT=str(Path(self.env['DATA']) / first))
            self.assertEqual(result.returncode, 23)
            self.assertEqual(len(self.records()), 1)

    def test_relative_paths_resolve_from_caller_and_name_defaults(self):
        env = dict(self.env)
        env.pop('MODEL_NAME')
        env.update(DATA='relative data', MODEL_PATH='relative model/', OUTPUT_DIR='relative results')
        result = subprocess.run(['/bin/bash', str(ROOT / 'eval.sh')], cwd=self.cwd,
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.records()[0]['args']
        self.assertEqual(args[args.index('--input_file') + 1], str(self.cwd / 'relative data' / 'valid.all.parquet'))
        self.assertEqual(args[args.index('--output_file') + 1], str(self.cwd / 'relative results' / 'relative model_math.jsonl'))

    def test_shell_syntax_and_strict_mode(self):
        for path in ROOT.glob('*.sh'):
            source = path.read_text()
            self.assertTrue(source.startswith('#!/usr/bin/env bash\nset -euo pipefail')
                            or 'set -euo pipefail' in source)
            self.assertNotIn('set -x', source)
            result = subprocess.run(['/bin/bash', '-n', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    unittest.main(verbosity=2)
