#!/usr/bin/env bash
set +x
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_common.sh"
init_experiment LUFFY_code_bs_128_32_TR_0.2_rb_cos_min_0.1_fill_off_clip_1.24 luffy-code Qwen2.5-Math-7B-16k-think

# Code-execution reward: configure compiler & timeouts
export LUFFY_CPP_COMPILER=g++
export LUFFY_CODE_COMPILE_TIMEOUT=20
# RUN_TIMEOUT is a CPU-TIME limit (enforced by prlimit --cpu on Linux), NOT
# wall-clock. The wall-clock deadline is RUN_TIMEOUT * WALL_MULTIPLIER so that
# CPU contention across parallel reward workers does not spuriously TLE correct
# programs (this was the cause of on_score collapsing under parallel reward).
export LUFFY_CODE_RUN_TIMEOUT=10
export LUFFY_CODE_WALL_MULTIPLIER=60
export LUFFY_CODE_MEMORY_LIMIT_MB=1024
# Keep -O2: competitive-programming solutions are tuned to the time limit.
export LUFFY_CODE_OPT_LEVEL=-O2
# cache (code, tests) -> verdict across the 8 GRPO samples and across steps
export LUFFY_CODE_CACHE_SIZE=50000
# reward threads. With CPU-time based timeouts this can exceed core count
# safely, but keep it moderate.
export LUFFY_REWARD_WORKERS=32

# Preserve the original algorithm and numerical hyperparameters.
run_training -m verl.mix_src.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    "data.train_files=$(hydra_string "$DATA_DIR/openr1-code-train.parquet")" \
    "data.val_files=$(hydra_string "$DATA_DIR/openr1-code-valid.parquet")" \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    "actor_rollout_ref.model.path=$(hydra_string "$MODEL_PATH")" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.max_prefix_len=8192 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger="[console,wandb]" \
    "trainer.project_name=$(hydra_string "$WANDB_PROJECT")" \
    "trainer.experiment_name=$(hydra_string "$EXP_NAME")" \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_sft_prefix_reward=False \
    actor_rollout_ref.rollout.enable_complete_with_prefix=True \
    actor_rollout_ref.rollout.enable_rollback=True \
    actor_rollout_ref.rollout.prefix_share_across_samples=True \
    actor_rollout_ref.rollout.prefix_strategy=cosine \
    actor_rollout_ref.rollout.prefix_linear_max_ratio=1.0 \
    actor_rollout_ref.rollout.prefix_linear_min_ratio=0.1 \
    actor_rollout_ref.rollout.n_prefix=1 \
    actor_rollout_ref.rollout.prefix_steps=500 \
    actor_rollout_ref.rollout.prefix_warmup_steps=50 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.actor.clip_upper_bound=1.2 \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_reshape="scaled_tr" \
    actor_rollout_ref.actor.off_policy_scaled_tr_coeff=0.2 \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    actor_rollout_ref.actor.expand_prefix_mask_to_sample=True \
    algorithm.grpo_use_std=False \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=False \
    data.reward_impl_version=3 \
    trainer.max_optim_to_keep=2 \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    "trainer.default_local_dir=$(hydra_string "$OUTPUT_DIR")" \
    trainer.total_epochs=30 "$@"
