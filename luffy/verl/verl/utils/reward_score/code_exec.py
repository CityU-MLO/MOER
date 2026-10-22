"""Local C++17 code-execution reward.

The model output is expected to contain some reasoning text followed by a
```cpp ... ``` block. The code reads from stdin and writes to stdout. We extract
the code block, compile it with a local C++17 compiler and run it against every
test case of the sample. The reward is ``1.0`` only if **all** test cases pass,
otherwise ``0.0``.

Design follows the local-execution style of verl's ``prime_code`` reward
(the remote alternative is ``sandbox_fusion``); everything here is local so no
service needs to be deployed.

Everything is configurable through environment variables so that no yaml change
is required:

==========================  ======================================  ========
env var                     meaning                                 default
==========================  ======================================  ========
LUFFY_CPP_COMPILER          compiler binary                         g++
LUFFY_CODE_COMPILE_TIMEOUT  compile timeout (seconds)               20.0
LUFFY_CODE_RUN_TIMEOUT      per-test-case timeout (seconds)         6.0
LUFFY_CODE_MEMORY_LIMIT_MB  per-process address-space limit (MB)    1024
LUFFY_CODE_FLOAT_TOL        tolerance when comparing numbers        1e-6
LUFFY_CODE_MAX_TESTS        max #test cases per sample (0 = all)    0
==========================  ======================================  ========
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple


# NOTE: We avoid `import math` because the sibling math.py (reward scoring for
# math problems) shadows the stdlib math module when this file is imported from
# within the same package. We only need isnan/isinf, implemented inline.
def _isnan(x: float) -> bool:
    return x != x


def _isinf(x: float) -> bool:
    return not _isnan(x) and _isnan(x - x)

CPP_COMPILER = os.environ.get("LUFFY_CPP_COMPILER", "g++")
COMPILE_TIMEOUT = float(os.environ.get("LUFFY_CODE_COMPILE_TIMEOUT", 20.0))
RUN_TIMEOUT = float(os.environ.get("LUFFY_CODE_RUN_TIMEOUT", 6.0))
MEMORY_LIMIT_MB = int(os.environ.get("LUFFY_CODE_MEMORY_LIMIT_MB", 1024))
FLOAT_TOL = float(os.environ.get("LUFFY_CODE_FLOAT_TOL", 1e-6))
MAX_TESTS = int(os.environ.get("LUFFY_CODE_MAX_TESTS", 0))
# When prlimit enforces the CPU-time limit, the wall-clock deadline is set to
# RUN_TIMEOUT * this multiplier so that CPU contention under many parallel
# reward workers does not spuriously time-out correct programs.
WALL_CLOCK_MULTIPLIER = float(os.environ.get("LUFFY_CODE_WALL_MULTIPLIER", 10.0))
# Optimisation level for the reward compiler. Competitive-programming
# solutions are often written right at the time limit, so -O0 makes many
# otherwise-correct programs TLE. Keep -O2 (the original behaviour) by default.
OPT_LEVEL = os.environ.get("LUFFY_CODE_OPT_LEVEL", "-O2")
# Cache compile+run verdicts keyed by (code_hash, test_cases_hash). GRPO samples
# the same prompt n times, so identical programs show up repeatedly both within
# one batch and across steps.
CACHE_SIZE = int(os.environ.get("LUFFY_CODE_CACHE_SIZE", 20000))

_RESULT_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    with _CACHE_LOCK:
        hit = _RESULT_CACHE.get(key)
        if hit is not None:
            _RESULT_CACHE.move_to_end(key)
            return dict(hit)
    return None


def _cache_put(key: str, value: Dict[str, Any]) -> None:
    with _CACHE_LOCK:
        _RESULT_CACHE[key] = dict(value)
        _RESULT_CACHE.move_to_end(key)
        while len(_RESULT_CACHE) > CACHE_SIZE:
            _RESULT_CACHE.popitem(last=False)

# ``` cpp | c++ | C++ | cc ``` ... ```
_CODE_BLOCK_RE = re.compile(
    r"```(?:[ \t]*)(cpp|c\+\+|cxx|cc|c)?[ \t]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# 1. extract the ```cpp ... ``` block from the model output
# --------------------------------------------------------------------------- #
def extract_cpp_code(model_output: str) -> Optional[str]:
    """Return the C++ code inside the last ```cpp ... ``` block, else ``None``.

    The model answer format is::

        <reasoning ...>
        ```cpp
        #include <iostream>
        ...
        ```

    Rules:
      * prefer the *last* fenced block explicitly tagged as cpp/c++ (that is the
        final solution),
      * fall back to the last untagged fenced block that looks like C++,
      * fall back to the raw text if it already looks like a C++ program.
    """
    if not model_output:
        return None

    tagged, untagged = [], []
    for lang, body in _CODE_BLOCK_RE.findall(model_output):
        if not body.strip():
            continue
        if lang and lang.lower() in ("cpp", "c++", "cxx", "cc", "c"):
            tagged.append(body)
        else:
            untagged.append(body)

    if tagged:
        return tagged[-1].strip("\n")

    for body in reversed(untagged):
        if _looks_like_cpp(body):
            return body.strip("\n")

    # unterminated final block, e.g. the generation was truncated
    idx = model_output.rfind("```cpp")
    if idx != -1:
        body = model_output[idx + len("```cpp"):]
        body = body.split("```")[0]
        if _looks_like_cpp(body):
            return body.strip("\n")

    if _looks_like_cpp(model_output):
        return model_output
    return None


def _looks_like_cpp(text: str) -> bool:
    return ("int main" in text) or ("#include" in text)


# --------------------------------------------------------------------------- #
# 2. output comparison (whitespace insensitive, float tolerant)
# --------------------------------------------------------------------------- #
def _split_lines(text: str) -> List[str]:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    while lines and lines[0] == "":
        lines.pop(0)
    return lines


def _tokens_match(got: Sequence[str], exp: Sequence[str], tol: float) -> bool:
    if len(got) != len(exp):
        return False
    for x, y in zip(got, exp):
        if x == y or x.lower() == y.lower():  # handles YES / Yes / yes
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if _isnan(fx) or _isnan(fy) or _isinf(fx) != _isinf(fy):
            return False
        diff = abs(fx - fy)
        if diff <= tol or diff <= tol * max(abs(fx), abs(fy)):
            continue
        return False
    return True


def outputs_match(actual: str, expected: str, tol: float = FLOAT_TOL) -> bool:
    """Compare program output with the reference output."""
    got_lines, exp_lines = _split_lines(actual), _split_lines(expected)
    if got_lines == exp_lines:
        return True
    if len(got_lines) == len(exp_lines):
        return all(
            _tokens_match(g.split(), e.split(), tol)
            for g, e in zip(got_lines, exp_lines)
        )
    # line breaks may differ; fall back to a flat token comparison
    return _tokens_match(actual.split(), expected.split(), tol)


# --------------------------------------------------------------------------- #
# 3. compile & run
# --------------------------------------------------------------------------- #
# Whether a usable `prlimit` binary exists. We use it to apply CPU/memory/core
# limits from OUTSIDE the child process, which is thread-safe. A preexec_fn that
# calls resource.setrlimit is NOT safe in a multithreaded process (Python docs:
# "it is not safe to use preexec_fn in the presence of threads") and under a
# ThreadPoolExecutor it can make child processes fail spuriously -> correct
# programs get scored 0. This bug only surfaced for on-policy samples (unique
# programs that actually run) while off-policy samples hit the result cache and
# never spawned a subprocess.
_PRLIMIT = shutil.which("prlimit")


def _resource_limit_prefix(memory_limit_mb: int, cpu_seconds: float) -> List[str]:
    """Build a `prlimit ...` command prefix; empty list if prlimit unavailable."""
    if not _PRLIMIT or os.name != "posix":
        return []
    cpu = int(cpu_seconds) + 1
    prefix = [_PRLIMIT, f"--cpu={cpu}", "--core=0"]
    if memory_limit_mb and memory_limit_mb > 0:
        nbytes = memory_limit_mb * 1024 * 1024
        prefix.append(f"--as={nbytes}")
    prefix.append("--")
    return prefix


def compile_cpp(
    code: str,
    workdir: str,
    compiler: str = CPP_COMPILER,
    timeout: float = COMPILE_TIMEOUT,
) -> Tuple[Optional[str], str]:
    """Compile ``code`` inside ``workdir``; return ``(binary_path, stderr)``."""
    src = os.path.join(workdir, "main.cpp")
    binary = os.path.join(workdir, "main.bin")
    with open(src, "w", encoding="utf-8") as f:
        f.write(code)

    cmd = [compiler, "-std=c++17", OPT_LEVEL, "-pipe", "-w", "-o", binary, src]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "compile timeout"
    except FileNotFoundError:
        return None, f"compiler not found: {compiler}"

    if proc.returncode != 0 or not os.path.exists(binary):
        return None, proc.stderr.decode("utf-8", errors="ignore")[-4000:]
    return binary, ""


def run_binary(
    binary: str,
    stdin_text: str,
    timeout: float = RUN_TIMEOUT,
    memory_limit_mb: int = MEMORY_LIMIT_MB,
    cwd: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Run ``binary`` feeding ``stdin_text``; return ``(status, stdout, err)``.

    ``status`` is one of ``ok`` / ``timeout`` / ``runtime_error``.
    """
    data = stdin_text if stdin_text.endswith("\n") else stdin_text + "\n"
    # Apply resource limits via `prlimit` from the parent side (thread-safe),
    # NOT via preexec_fn which is unsafe under threads.
    #
    # IMPORTANT: `timeout` is a CPU-time limit, enforced by `prlimit --cpu`.
    # The wall-clock timeout given to subprocess.run must be much larger,
    # otherwise CPU contention under a ThreadPoolExecutor (N subprocesses
    # sharing < N cores) makes correct programs exceed the wall-clock deadline
    # and be scored 0. The wall-clock deadline here is only a safety net for
    # processes that block on I/O / deadlock and never burn CPU.
    limit_prefix = _resource_limit_prefix(memory_limit_mb, timeout)
    cmd = limit_prefix + [binary]
    if limit_prefix:
        # prlimit enforces the real (CPU-time) limit; give wall-clock lots of slack
        wall_timeout = max(timeout * WALL_CLOCK_MULTIPLIER, timeout + 30.0)
    else:
        # no prlimit available (e.g. macOS dev box): fall back to wall-clock only
        wall_timeout = timeout
    try:
        proc = subprocess.run(
            cmd,
            input=data.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=wall_timeout,
            cwd=cwd or os.path.dirname(binary),
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        return "timeout", "", f"timeout after {wall_timeout}s (wall)"
    except Exception as exc:  # noqa: BLE001
        return "runtime_error", "", repr(exc)

    # prlimit kills a CPU-time overrun with SIGKILL (-9) / SIGXCPU (-24)
    if limit_prefix and proc.returncode in (-9, -24, 128 + 9, 128 + 24):
        return "timeout", "", f"CPU time limit exceeded ({int(timeout) + 1}s)"

    stdout = proc.stdout.decode("utf-8", errors="ignore")
    stderr = proc.stderr.decode("utf-8", errors="ignore")[-2000:]
    if proc.returncode != 0:
        return "runtime_error", stdout, f"exit code {proc.returncode}: {stderr}"
    return "ok", stdout, stderr


# --------------------------------------------------------------------------- #
# 4. test cases normalisation
# --------------------------------------------------------------------------- #
def normalize_test_cases(value: Any) -> List[Dict[str, str]]:
    """Coerce anything stored in ``ground_truth`` into ``[{input, output}, ...]``."""
    if value is None:
        return []

    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            return []

    if isinstance(value, dict):
        for key in ("examples", "test_cases", "tests", "cases"):
            if key in value:
                return normalize_test_cases(value[key])
        if "input" in value or "output" in value:
            value = [value]
        elif "inputs" in value and "outputs" in value:
            return [
                {"input": str(i), "output": str(o)}
                for i, o in zip(value["inputs"], value["outputs"])
            ]
        else:
            return []

    cases: List[Dict[str, str]] = []
    try:
        iterator = list(value)
    except TypeError:
        return []
    for item in iterator:
        if isinstance(item, dict):
            if "input" not in item and "output" not in item:
                continue
            cases.append(
                {
                    "input": "" if item.get("input") is None else str(item.get("input")),
                    "output": "" if item.get("output") is None else str(item.get("output")),
                }
            )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            cases.append({"input": str(item[0]), "output": str(item[1])})
    return cases


# --------------------------------------------------------------------------- #
# 5. public evaluation API
# --------------------------------------------------------------------------- #
def evaluate_cpp_code(
    code: str,
    test_cases: Sequence[Dict[str, str]],
    run_timeout: float = RUN_TIMEOUT,
    compile_timeout: float = COMPILE_TIMEOUT,
    memory_limit_mb: int = MEMORY_LIMIT_MB,
    float_tol: float = FLOAT_TOL,
    early_stop: bool = True,
) -> Dict[str, Any]:
    """Compile once, run every test case; return a detailed result dict."""
    code_hash = hashlib.md5(code.encode("utf-8")).hexdigest() if code else ""
    result: Dict[str, Any] = {
        "passed": False,
        "num_total": len(test_cases),
        "num_passed": 0,
        "status": "ok",
        "error": "",
        "code_hash": code_hash,
    }
    if not code:
        result.update(status="no_code", error="no cpp code block found")
        return result
    if not test_cases:
        result.update(status="no_test_case", error="no test case available")
        return result

    # identical (program, tests) pair -> reuse the previous verdict
    cache_key = code_hash + ":" + hashlib.md5(
        json.dumps(list(test_cases), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    workdir = tempfile.mkdtemp(prefix="luffy_cpp_")
    try:
        binary, err = compile_cpp(code, workdir, timeout=compile_timeout)
        if binary is None:
            result.update(status="compile_error", error=err)
            _cache_put(cache_key, result)
            return result

        for idx, case in enumerate(test_cases):
            status, stdout, err = run_binary(
                binary,
                case.get("input", ""),
                timeout=run_timeout,
                memory_limit_mb=memory_limit_mb,
                cwd=workdir,
            )
            if status != "ok":
                result.update(status=status, error=f"test {idx}: {err}")
                if early_stop:
                    _cache_put(cache_key, result)
                    return result
                continue
            if outputs_match(stdout, case.get("output", ""), tol=float_tol):
                result["num_passed"] += 1
            else:
                result.update(
                    status="wrong_answer",
                    error=f"test {idx}: expected {case.get('output', '')!r}, got {stdout!r}",
                )
                if early_stop:
                    _cache_put(cache_key, result)
                    return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    result["passed"] = result["num_passed"] == result["num_total"]
    if result["passed"]:
        result["status"] = "ok"
    _cache_put(cache_key, result)
    return result


def evaluate_model_output(
    model_output: str,
    test_cases: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Extract the cpp block from ``model_output`` then evaluate it."""
    cases = normalize_test_cases(test_cases)
    code = extract_cpp_code(model_output)
    if code is None:
        return {
            "passed": False,
            "num_total": len(cases),
            "num_passed": 0,
            "status": "no_code",
            "error": "no cpp code block found",
            "code_hash": "",
        }
    return evaluate_cpp_code(code, cases, **kwargs)


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
    **kwargs: Any,
) -> float:
    """Reward interface used by ``RewardManager``: ``1.0`` if all tests pass.

    Also enforces the <think>...</think> format requirement:
    the model output must contain both ``<think>`` and ``</think>`` tags,
    otherwise the reward is 0.
    """
    if not solution_str:
        return 0.0

    # Format check: must have <think>...</think>
    if "<think>" not in solution_str or "</think>" not in solution_str:
        return 0.0

    # Extract the part after </think> for evaluation
    solution_str = solution_str.split("</think>", 1)[1]

    if (ground_truth is None or (hasattr(ground_truth, '__len__') and len(ground_truth) == 0)) and isinstance(extra_info, dict):
        ground_truth = extra_info.get("examples") or extra_info.get("test_cases")
    try:
        info = evaluate_model_output(solution_str, ground_truth, **kwargs)
    except Exception as exc:  # noqa: BLE001 - never crash the training loop
        print(f"[code_exec] unexpected error: {exc!r}", file=sys.stderr)
        return 0.0
    return 1.0 if info["passed"] else 0.0


# backwards/naming compatible aliases
reward_fn_code_exec = compute_score


if __name__ == "__main__":
    # NOTE: compute_score requires the model output to contain <think>...</think>
    # (format reward), and evaluates only the text AFTER </think>.
    demo_output = """<think>
We just need to read the two integers and print their sum.
</think>

Here is the solution.

```cpp
#include <iostream>
int main() {
    long long a, b;
    std::cin >> a >> b;
    std::cout << a + b << std::endl;
    return 0;
}
```"""
    cases = [{"input": "1 2", "output": "3"}, {"input": "10 -4", "output": "6"}]
    print("reward(correct) =", compute_score(demo_output, cases))
    print("reward(wrong gt) =", compute_score(demo_output, [{"input": "1 2", "output": "4"}]))
    print("reward(no code) =", compute_score("<think>x</think> just thinking, no code", cases))
    print("reward(no think) =", compute_score(demo_output.replace("<think>", "").replace("</think>", ""), cases))
    print("float tolerance:", outputs_match("2.666666666666667\n", "2.6666666666666665"))
