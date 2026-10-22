# MOER

**Mix-Policy Optimization under Expert Prefix Guidance with Selective Rollback**

Research implementation accompanying the MOER manuscript. MOER combines decaying expert guidance with model exploration to improve mathematical, general-domain, and code reasoning. The implementation builds on LUFFY and verl; historical `luffy`, `deepscaler`, and `verl` package names are retained for compatibility.

This repository does not announce an accepted venue, public MOER checkpoint, or public paper URL. The manuscript results below are reported results, not measurements reproduced during repository cleanup.

## Method

MOER combines three components:

1. **Cosine-annealed expert prefix guidance.** Supply an expert chain-of-thought prefix and let the model generate the remaining reasoning. A warm-up phase uses full guidance, followed by a progressively shorter prefix.
2. **Selective rollout rollback.** Replace a failed guided completion with an already available full expert trace, without another model-generation call. Correctness relies on the reference trace and the chosen verifier; rollback is not an independent proof of correctness.
3. **Harmonized mix-policy objective.** Use a clipped GRPO objective for unguided on-policy tokens and trust-region supervision for guided trajectories. In the main MOER configuration, both prefix and completion tokens use the off-policy objective.

### Implementation map

Paths below are relative to `luffy/verl/verl/mix_src/`:

| Component | Implementation |
| --- | --- |
| Hydra/Ray entry point and reward routing | `main_mix_ppo.py` |
| Prefix schedule and guided generation | `mix_self_vllm_rollout.py` |
| Full-trace and other prefix baselines | `mix_vllm_rollout.py` |
| Rollback and mixed-batch training | `mix_trainer.py` |
| Harmonized token objectives | `mix_core_alg.py`, `mix_actor.py` |
| FSDP actor/rollout workers | `mix_fsdp_worker.py` |
| Expert trace dataset | `rl_dataset_with_target.py` |
| Default training configuration | `config/mix_ppo_trainer.yaml` |

With multiple prefix candidates, the current rollback implementation replaces the first candidate only when all prefix candidates in its prompt group fail. Main MOER uses one prefix candidate. Pure on-policy correctness is not part of this rollback condition.

## Repository layout

```text
.
├── README.md
├── start.sh                    # Offline installed-version check; does not install
├── data/                       # Dataset conversion and preparation scripts
├── exp_scripts/                # MOER, baselines, and ablation launchers
├── eval_scripts/               # Math, general reasoning, code, and pass@k evaluation
├── luffy/
│   ├── deepscaler/              # Retained reward utilities
│   ├── requirements.txt        # Curated common dependency pins
│   ├── requirements.v2.txt     # Compatibility filename with the same pins
│   └── verl/verl/
│       ├── mix_src/             # Core MOER implementation
│       ├── trapo_src/           # Retained TRAPO baseline
│       └── ...                  # Required upstream workers and utilities
├── tests/test_release.py       # CPU-only launch and release regression tests
└── tools/release.py            # Read-only release audit and source-only ZIP builder
```

Raw data, checkpoints, generated samples, credentials, IDE metadata, and assistant histories are deliberately excluded from the source release. They may still exist in a development working copy. Upstream licenses, notices, and author attribution are retained.

## Environment

Training and inference require **Linux, NVIDIA GPUs, and a compatible CUDA toolchain**. The core stack is Python 3.11, PyTorch 2.4.0, vLLM 0.6.3, TensorDict 0.5.0, Transformers 4.47.1, and Ray 2.38.0. FlashAttention 2.7.3 must match Python, PyTorch, CUDA, and the C++ ABI.

The two historical dependency snapshots have been condensed into a common set of direct requirements. They are not a complete transitive lockfile, and a clean Linux/GPU installation still needs validation. Do not mix the old snapshots or upgrade vLLM independently: the bundled integration is version-specific.

From the repository root, in a fresh, dedicated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r luffy/requirements.txt
python -m pip install flash-attn==2.7.3 --no-build-isolation
python -m pip install -e ./luffy -e ./luffy/verl
python -m pip check
PYTHON=python bash start.sh
```

For offline clusters, install from a reviewed wheelhouse instead. FlashAttention may need a prebuilt compatible wheel rather than a source build. `start.sh` only checks installed versions of the curated requirements; it does not install packages, configure proxies, start Ray, or validate CUDA kernels. FlashAttention and actual GPU readiness must be checked separately.

The editable installs include all `deepscaler` subpackages. Both verl installation metadata paths use the same dependency declarations and include MIX/TRAPO YAML configuration files.

## Data preparation

### Mathematics and general reasoning

The manuscript uses **OpenR1-Math-46k-8192**, with expert traces generated by DeepSeek-R1. Prepare its training parquet with:

```bash
python data/prepare_train.py --output data/openr1.parquet
```

This command downloads the public dataset when executed; no dataset is downloaded merely by importing the script. Acquire benchmark files separately and verify their provenance and licenses.

Expected files:

| Use | Filename |
| --- | --- |
| Main math training | `openr1.parquet` |
| Training-time validation | `valid.parquet` |
| Math evaluation suite | `valid.all.parquet` |
| General-domain evaluation | `valid.arc_c.parquet`, `valid.gpqa.parquet`, `valid.mmlu_pro.parquet` |
| AIME pass@k, 256 repeated rows per problem | `valid_aime_256.parquet`, `valid_aime25_256.parquet` |
| Held-out RL after SFT | `openr1.complement.parquet` |

Math training rows contain `prompt` and `target` chat-message lists, `data_source`, and verifier-compatible `reward_model` metadata. `prepare_train_sft_rl.py` prepares the complementary RL data; `prepare_sft.py` converts local traces to SFT JSONL. SFT-then-RL expects an already SFT-trained model; its launcher does not perform the preceding SFT stage.

### Code generation

The manuscript describes **CodeForces-CoTs**, filtering prompts over 1,024 tokens and responses over 8,192 tokens, with 13,000 training and 1,402 validation examples. Code rows use `prompt`, expert `target`, `data_source`, and public test cases in `examples`.

```bash
export LUFFY_TOKENIZER_PATH="/path/to/local/tokenizer"
python data/prepare_openr1_code.py \
  --input_dir /path/to/raw-code-parquet-shards \
  --output data/openr1-code.parquet \
  --tokenizer "$LUFFY_TOKENIZER_PATH"
python data/prepare_livecodebench.py \
  --input /path/to/livecodebench.jsonl \
  --output data/livecodebench.parquet \
  --tokenizer "$LUFFY_TOKENIZER_PATH"
```

The conversion helpers do **not** establish the manuscript's exact train/validation split. Supply the original split as `openr1-code-train.parquet` and `openr1-code-valid.parquet`, with recorded sample identifiers and preprocessing settings. `filter_by_gt_reward.py` currently checks whether reference C++ can be extracted; its name must not be interpreted as full execution-based correctness filtering. Some preprocessing helpers rewrite their input parquet, so use reviewed copies.

Verify that grading test cases are absent from model input prompts before training, as required by the manuscript's anti-reward-hacking setup. Extracting an `examples` column alone does not guarantee their removal from the prompt. The existing conversion code and experimental data were not silently rewritten to change this behavior.

The local LiveCodeBench converter scores public stdin/stdout C++ test cases and drops unsupported empty/non-stdin test cases. This is not a claim of equivalence to an official hidden-test LiveCodeBench score. The coding result table in the supplied manuscript is unfilled, so no coding accuracy numbers are asserted here.

## Training

All ten launchers locate the repository from their own path and accept environment configuration. Relative user-supplied paths are resolved against the caller's working directory; absolute paths are recommended.

```bash
export DATA_DIR="/path/to/prepared-data"
export MODEL_PATH="/path/to/local/model"
export OUTPUT_DIR="/path/to/training-results/moer"
export WANDB_MODE=offline
export WANDB_PROJECT=MOER
export EXP_NAME=moer-math-7b
export PYTHON=python

bash exp_scripts/train.sh \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  trainer.save_freq=50
```

Trailing arguments are Hydra overrides and take precedence over launcher defaults. The manuscript specifies an entropy coefficient of 0.01; the original launchers use 0.001. This difference is exposed rather than silently changed. Main launchers otherwise retain the supplied experimental hyperparameters. Several have `save_freq=-1`, which disables checkpoint saving; override it when a saved model is required.

Other environment controls: `LOG_DIR`, `WANDB_DIR`, `VERL_ROOT`, `PYTHON_BIN`, and `STOP_RAY`. Ray is stopped only with explicit `STOP_RAY=1`; never enable that on a shared node unless you own its Ray processes. Online logging requires `WANDB_MODE=online` and an externally supplied `WANDB_API_KEY`. Do not put credentials into scripts, command-line arguments, or tracked files.

| Launcher | Purpose |
| --- | --- |
| `train.sh` | MOER: cosine prefix, rollback, and guided-completion off-policy loss |
| `train_code.sh` | MOER code-generation experiment; supply the instruction-tuned coding model |
| `train_only_loss.sh` | Objective-only/full-expert-guidance ablation |
| `train_on_policy.sh` | On-policy GRPO baseline |
| `train_luffy.sh` | LUFFY full-expert-trace baseline |
| `train_chord.sh` | CHORD-style RL/SFT baseline |
| `train_prefixrft.sh` | Prefix-RFT variant with high-entropy prefix selection in the optimized loss |
| `train_trapo.sh` | TRAPO baseline using `verl.trapo_src` |
| `train_rl_sft_loss.sh` | RL with an auxiliary SFT loss |
| `train_sft_rl.sh` | RL stage after SFT, using complementary data |

The original math tokenizer/model names include a `<think>` chat-template convention. Use a compatible local model/tokenizer and verify its template. For the coding experiment, explicitly point `MODEL_PATH` to the intended Qwen2.5-Instruct-7B-compatible local model rather than relying on the historical launcher default.

### Main manuscript settings

| Setting | Value / configuration |
| --- | --- |
| Rollout/update batch size | 128 / 64 |
| Group size; prefix candidates | 8; 1 |
| Prefix warm-up; annealing horizon | 50 steps; launcher horizon 500 steps |
| Prefix ratio range | 1.0 to 0.1 |
| GRPO lower/upper clipping | 0.2 / 0.24; upper ratio bound 1.24 |
| Advantage standard-deviation normalization | Disabled |
| KL penalty | Disabled |
| Off-policy trust-region floor | 0.1 in the current `scaled_tr` implementation |
| Off-policy scale | 0.2 for math models; use 0.1 for the instruction model where specified |

The off-policy scale is `actor_rollout_ref.actor.off_policy_scaled_tr_coeff`. Match the exact model, data split, template, verifier, seed, and overrides before interpreting a run as a manuscript reproduction.

## Evaluation

```bash
export DATA="/path/to/benchmark-parquet-directory"
export MODEL_PATH="/path/to/local/huggingface-format-checkpoint"
export OUTPUT_DIR="/path/to/evaluation-results"
export MODEL_NAME=moer
export GPU=0,1,2,3
export PYTHON=python

bash eval_scripts/eval.sh
bash eval_scripts/pass_at_k.sh
bash eval_scripts/eval_livecodebench.sh
```

The evaluation tools use local Hugging Face-format models, not raw FSDP shards. Prepare/export that model separately. `eval_livecodebench.sh` also runs ARC-c, GPQA, and MMLU-Pro, so all four corresponding inputs are required.

Use `--max_tokens 4096` as a trailing override when needed. `TEMPLATE` may explicitly select `own`, `qwen`, `prime`, or another supported template. Ordinary evaluation generates one sample per input row. AIME pass@k infers 256 samples from the original repeated-row inputs; it does not turn an ordinary unreplicated parquet into 256 samples automatically.

The manuscript reports avg@32 on AIME 2024/2025 and AMC, and pass@1 on the other math benchmarks and general-domain benchmarks. Ensure repeated inputs for `valid.all.parquet` match that protocol; this is distinct from the 256-sample pass@k experiment. The mathematical suite contains AIME 2024, AIME 2025, AMC, MATH-500, Minerva, and OlympiadBench.

By default, prompts are neither printed nor stored in evaluation output. Explicit `--include_prompt True` enables storage and `--print_sample True` enables sample printing. Generated text and answers can still be sensitive. Do not publish result JSONL without review. Cache validation checks input identity and sample counts, but use separate output paths or force regeneration when changing model weights or sampling settings.

**Code-execution safety:** `code_exec.py` compiles and executes untrusted generated C++ locally. CPU/memory limits and a temporary working directory are **not a security sandbox**. Run code evaluation only inside a dedicated, non-privileged, secret-free, network-isolated environment with appropriate filesystem/process controls. Do not run it on a machine holding personal files or credentials.

## Acknowledgments

This implementation builds on LUFFY, verl, DeepScaler, vLLM, and Math-Verify.
