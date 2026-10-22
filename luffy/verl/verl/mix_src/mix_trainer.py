# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from collections import defaultdict, Counter

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import torch

from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    ResourcePoolManager,
    WorkerType,
    _timer,
    # compute_data_metrics,
    compute_timing_metrics,
    dataprotoitem_to_dataproto,
    # compute_advantage,
    reduce_metrics
)
from verl.utils.torch_functional import masked_mean


def compute_prefix_replace_mask(score: torch.Tensor, n: int, n_prefix: int):
    """Return the rollback replacement masks used in mix training.

    score contains [bsz] scores from interleaving n samples per prompt. The
    first n_prefix positions in each group are prefix candidates. Replace
    only the first prefix candidate when every prefix candidate in its group
    scores below 1. No replacement occurs when n_prefix is 0. The correctness
    of non-prefix candidates does not participate in the current decision.

    Return (replace_mask, prefix_candidate_mask), both [bsz] boolean tensors
    on the score device. The caller ensures bsz is divisible by n and
    0 <= n_prefix <= n.
    """
    bsz = score.shape[0]
    device = score.device

    # positions whose within-group index < n_prefix
    group_idx = torch.arange(bsz, device=device) % n
    prefix_candidate_mask = (group_idx < n_prefix)  # [bsz], all prefix-slot samples

    # First-prefix-only rule (replaces the previous `score < 1` condition):
    # Within each group, only when ALL prefix-slot samples ([:n_prefix]) are
    # incorrect (score < 1), mark ONLY the first prefix sample as True; all
    # other prefix samples stay False.
    if n_prefix > 0:
        num_groups = bsz // n
        score_grouped_full = score.view(num_groups, n)  # [G, n]
        prefix_all_wrong = (score_grouped_full[:, :n_prefix] < 1).all(dim=1)  # [G]
        # Build a per-sample mask that is True only at the first prefix slot
        # of each group, and only when that group's prefix samples are all wrong.
        first_prefix_only = torch.zeros(num_groups, n, dtype=torch.bool, device=device)
        first_prefix_only[:, 0] = prefix_all_wrong
        first_prefix_only_per_sample = first_prefix_only.view(bsz)
    else:
        first_prefix_only_per_sample = torch.zeros_like(prefix_candidate_mask)

    replace_mask = (
        prefix_candidate_mask
        & first_prefix_only_per_sample
    )  # [bsz] bool on score.device

    return replace_mask, prefix_candidate_mask


# directly copied from verl/trainer/ppo/ray_trainer.py
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics

def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, grpo_use_std=True):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        use_std=grpo_use_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo_split':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        prefix_mask = data.batch['prefix_mask']
        on_policy_mask = ~prefix_mask.any(-1)
        from .mix_core_alg import compute_grpo_outcome_advantage_split
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            on_policy_mask=on_policy_mask,
            use_std=grpo_use_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns

    elif adv_estimator == 'reinforce':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                             eos_mask=response_mask,
                                                                             index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data

class MIXRayPPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'grpo_split':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from torch.utils.data import DataLoader, SequentialSampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        from .rl_dataset_with_target import RLHFDatasetWithTarget
        self.train_dataset = RLHFDatasetWithTarget(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True, return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         max_target_length=self.config.actor_rollout_ref.rollout.max_prefix_len,
                                         filter_targets=self.config.data.get('filter_targets', False),
                                         sample_target_ratio=self.config.data.get('sample_target_ratio', 1.0))

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            from verl.mix_src.rl_dataset_with_target import ResumableRandomSampler
            sampler = ResumableRandomSampler(data_source=self.train_dataset)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _save_checkpoint(self):
        # Override the base implementation locally to support samplers without get_state.
        import dill
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps)
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps)

        torch.save(self.train_dataloader, os.path.join(local_global_step_folder, 'data.pt'), pickle_module=dill)
        sampler = getattr(self.train_dataloader, 'sampler', None)
        if callable(getattr(sampler, 'get_state', None)):
            torch.save(sampler.get_state(), os.path.join(local_global_step_folder, 'sampler_state.pt'),
                       pickle_module=dill)
        with open(os.path.join(self.config.trainer.default_local_dir,
                               'latest_checkpointed_iteration.txt'), 'w') as f:
            f.write(str(self.global_steps))

        # Preserve the checkpoint cleanup policy of the base implementation.
        try:
            cur_step = self.global_steps
            prev_save_steps = list(range(0, cur_step, self.config.trainer.save_freq))
            prev_save_steps = prev_save_steps[:-(self.config.trainer.max_optim_to_keep - 1)]
            for each_save_step in prev_save_steps:
                prev_save_path = os.path.join(self.config.trainer.default_local_dir,
                                              f'global_step_{each_save_step}', 'actor')
                if not os.path.exists(prev_save_path):
                    continue
                for path in os.listdir(prev_save_path):
                    if (path.startswith('optim') or path.startswith('model_world_size')) and path.endswith('.pt'):
                        os.remove(os.path.join(prev_save_path, path))
                        print(f'Removed {os.path.join(prev_save_path, path)}')
        except Exception as e:
            print(f'Error removing previous checkpoints: {e}')

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')

        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
        checkpoint_folder = os.path.abspath(self.config.trainer.default_local_dir)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        elif not (self.config.trainer.resume_from_path and global_step_folder is not None):
            assert isinstance(self.config.trainer.resume_mode, str), 'resume ckpt must be str type'
            assert 'global_step_' in self.config.trainer.resume_mode, 'resume ckpt must specify the global_steps'
            global_step_folder = os.path.abspath(self.config.trainer.resume_mode)

        self.global_steps = int(global_step_folder.split('global_step_')[-1])
        print(f'Resuming from {global_step_folder}, global_step={self.global_steps}')
        self.actor_rollout_wg.load_checkpoint(os.path.join(global_step_folder, 'actor'))
        if self.use_critic:
            self.critic_wg.load_checkpoint(os.path.join(global_step_folder, 'critic'))

        import dill
        from verl.utils.dataset.rl_dataset import RLHFDataset
        # data.pt is a trusted local DataLoader checkpoint; restore the complete object.
        self.train_dataloader = torch.load(os.path.join(global_step_folder, 'data.pt'),
                                           pickle_module=dill, weights_only=False)
        self.train_dataset = self.train_dataloader.dataset
        if isinstance(self.train_dataset, RLHFDataset):
            self.train_dataset.resume_dataset_state()
        sampler = getattr(self.train_dataloader, 'sampler', None)
        sampler_state_path = os.path.join(global_step_folder, 'sampler_state.pt')
        if callable(getattr(sampler, 'set_state', None)) and os.path.isfile(sampler_state_path):
            sampler.set_state(torch.load(sampler_state_path, pickle_module=dill, weights_only=False))

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        last_validation_step = None
        last_checkpoint_step = None

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            last_validation_step = self.global_steps
            if self.config.trainer.get('val_only', False):
                return

        # Checkpoint steps count completed updates; increment only when starting the next step.
        if self.global_steps >= self.total_training_steps:
            return

        n_samples = self.config.actor_rollout_ref.rollout.n
        if self.config.data.get('add_tgt_with_acc', False):
            n_samples = n_samples - 1 # if filter tgt with acc, we either use tgt or on policy samples.

        for _ in range(self.config.trainer.total_epochs):

            for batch_dict in self.train_dataloader:
                self.global_steps += 1
                is_last_step = self.global_steps >= self.total_training_steps
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                metrics = {}
                timing_raw = {}

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids', 'tgt_cot_input_ids'])
                gen_batch.meta_info['global_steps'] = self.global_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    # This code matches a prompt ID with its N responses.
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    # log avg prefix ratio
                    if 'prefix_ratios' in gen_batch_output.meta_info.keys():
                        metrics['batch/avg_prefix_ratio'] = float(np.mean(gen_batch_output.meta_info['prefix_ratios']))

                    if self.config.trainer.add_full_target_when_none:
                        pass

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):

                        # Reward computed during the rollback pass. Rows that were NOT
                        # replaced keep this value; only replaced rows need recomputing.
                        reward_tensor_cached = None
                        replace_indices = None

                        if self.config.actor_rollout_ref.rollout.enable_complete_with_prefix and self.config.actor_rollout_ref.rollout.enable_rollback:
                            # for filtering incorrect prefix response
                            n = self.config.actor_rollout_ref.rollout.n
                            n_prefix = self.config.actor_rollout_ref.rollout.n_prefix
                            reward_tensor_tmp = self.reward_fn(batch)  # [bsz, l], only the last valid token has reward
                            score = reward_tensor_tmp.sum(-1)  # keep on device, shape [bsz]

                            bsz = score.shape[0]
                            replace_mask, prefix_candidate_mask = compute_prefix_replace_mask(
                                score=score, n=n, n_prefix=n_prefix,
                            )

                            # log replacement rate
                            num_replaced = int(replace_mask.sum().item())
                            num_prefix_candidates = int(prefix_candidate_mask.sum().item())
                            metrics['batch/replace_num'] = num_replaced
                            metrics['batch/replace_ratio_overall'] = (
                                num_replaced / bsz if bsz > 0 else 0.0
                            )
                            metrics['batch/replace_ratio_in_prefix_slots'] = (
                                num_replaced / num_prefix_candidates
                                if num_prefix_candidates > 0 else 0.0
                            )

                            # Reuse the reward we just computed; the rows that get
                            # overwritten below are recomputed afterwards.
                            reward_tensor_cached = reward_tensor_tmp
                            if replace_mask.any():
                                # plain python list of ints: safe for both TensorDict
                                # fancy-indexing and numpy object-array indexing
                                replace_indices = replace_mask.reshape(-1).nonzero(
                                    as_tuple=True
                                )[0].cpu().tolist()
                                # move mask to the same device as each target tensor to be safe
                                # Pull out the full tensors, modify out-of-place, then write back.
                                # This avoids relying on in-place row assignment through TensorDict's
                                # __getitem__ (which may return a view/copy depending on tensordict
                                # version / lazy settings and could silently drop the write).
                                for key, tgt_key in (
                                    ('input_ids',      'tgt_input_ids'),
                                    ('responses',      'tgt_responses'),
                                    ('attention_mask', 'tgt_attention_mask'),
                                    ('prefix_mask',    'tgt_prefix_mask'),
                                ):
                                    src = batch.batch[key]
                                    tgt = batch.batch[tgt_key]
                                    assert src.shape == tgt.shape, (
                                        f"shape mismatch when replacing with target: "
                                        f"{key}{tuple(src.shape)} vs {tgt_key}{tuple(tgt.shape)}"
                                    )
                                    mask = replace_mask.to(src.device)
                                    # cast in case dtypes differ (e.g. bool vs long)
                                    new_tensor = torch.where(
                                        mask.view(-1, *([1] * (src.ndim - 1))),
                                        tgt.to(dtype=src.dtype),
                                        src,
                                    )
                                    batch.batch[key] = new_tensor
                            batch.pop(batch_keys=['tgt_responses', 'tgt_attention_mask', 'tgt_prefix_mask'])

                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if reward_tensor_cached is None or self.use_rm:
                            # no rollback pass (or a reward model supplies rm_scores) -> full pass
                            reward_tensor = self.reward_fn(batch) # [bsz, l], only the last valid token has reward
                        elif replace_indices is None:
                            # nothing was replaced -> the cached reward is still exact
                            reward_tensor = reward_tensor_cached
                        else:
                            # only the replaced rows changed; recompute just those
                            reward_tensor = reward_tensor_cached.clone()
                            sub_batch = batch.slice(replace_indices)
                            sub_reward = self.reward_fn(sub_batch)  # [n_replaced, l]
                            idx = torch.as_tensor(
                                replace_indices, dtype=torch.long, device=reward_tensor.device
                            )
                            reward_tensor[idx] = sub_reward.to(
                                dtype=reward_tensor.dtype, device=reward_tensor.device
                            )
                        batch.batch['token_level_scores'] = reward_tensor

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)

                        if self.config.data.reward_impl_version == 0:
                            fail_value = 0
                            success_value = 1
                            format_value = -1 # not defined.
                        elif self.config.data.reward_impl_version == 1:
                            fail_value = -0.5
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 2:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 3:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 4:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        else:
                            raise ValueError(f'Invalid reward implementation version: {self.config.data.reward_impl_version}')

                        solve_none = 0
                        solve_all = 0
                        solve_none_format = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == fail_value).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards == success_value).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                            elif (uid_rewards == format_value).all():
                                valid_mask[uid_mask] = False
                                solve_none_format += 1

                        if self.config.trainer.skip_valid_mask:
                            valid_mask[:] = True
                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_none_format'] = solve_none_format
                        metrics['batch/solve_all'] = solve_all

                        # add more metrics
                        metrics['batch/solved'] = (reward_tensor.sum(-1) == success_value).sum().item() / len(uids)
                        metrics['batch/failed'] = (reward_tensor.sum(-1) == fail_value).sum().item() / len(uids)
                        # add on-policy metrics
                        prefix_mask = batch.batch['prefix_mask']
                        off_policy_mask = prefix_mask.any(-1)
                        on_policy_mask = ~off_policy_mask
                        metrics['batch/on_solved'] = (reward_tensor[on_policy_mask].sum(-1) == success_value).sum().item() / (on_policy_mask.sum().item() + 1e-6)
                        metrics['batch/off_solved'] = (reward_tensor[off_policy_mask].sum(-1) == success_value).sum().item() / (off_policy_mask.sum().item() + 1e-6)

                        # recompute old_log_probs
                        with _timer('old_log_prob', timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # NOTE: the advantages are the same for all tokens in the response
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  grpo_use_std=self.config.algorithm.grpo_use_std)

                        # compute alpha and beta for prefix reward weighting
                        prefix_mask = batch.batch['prefix_mask']
                        advantages = batch.batch['advantages']
                        assert prefix_mask.shape == advantages.shape

                        alpha_weight = prefix_mask.float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_alpha
                        beta_weight = (~prefix_mask).float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_beta
                        prefix_weight = alpha_weight + beta_weight
                        batch.batch['advantages'] = prefix_weight * advantages

                        if self.config.data.get('disable_truncation_advantage', False):
                            responses = batch.batch['responses']
                            responses_mask = responses != self.tokenizer.pad_token_id
                            response_length = responses_mask.sum(-1) # [bsz]
                            max_len = self.config.data.max_response_length
                            has_truncated = response_length >= max_len
                            no_eos = ~((responses == self.tokenizer.eos_token_id).any(-1))
                            truncated_mask = has_truncated & no_eos
                            batch.batch['advantages'][truncated_mask] = 0

                        if self.config.actor_rollout_ref.actor.get('use_sft_prefix_reward', False):
                            assert self.config.actor_rollout_ref.rollout.n_prefix == -1
                            reward_weight = self.config.actor_rollout_ref.actor.get('sft_prefix_reward_weight', 1.0)
                            batch.batch['advantages'][prefix_mask] = reward_weight / n_samples

                    if self.config.trainer.debug is True:
                        raise RuntimeError(f"trainer.debug=True: stopping before the update, global_step={self.global_steps}")

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and (
                        is_last_step or (self.config.trainer.test_freq > 0 and
                                         self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        if 'avg_score' not in val_metrics:
                            val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                        metrics.update(val_metrics)
                        self.maybe_save_best_hf(val_metrics)
                        last_validation_step = self.global_steps

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                        last_checkpoint_step = self.global_steps

                # collect metrics
                metrics.update(compute_data_metrics_ours(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    return

        # Validate and save the last completed update when epochs are exhausted.
        if self.val_reward_fn is not None and last_validation_step != self.global_steps:
            val_metrics = self._validate()
            if 'avg_score' not in val_metrics:
                val_metrics['avg_score'] = np.mean([value for key, value in val_metrics.items()
                                                   if key.startswith('val/test_score/')])
            self.maybe_save_best_hf(val_metrics)
            logger.log(data=val_metrics, step=self.global_steps)
        if self.global_steps > 0 and self.config.trainer.save_freq > 0 and last_checkpoint_step != self.global_steps:
            self._save_checkpoint()

    def maybe_save_best_hf(self, val_metrics: dict):
        import json
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'best',
                                        f'actor')

        os.makedirs(actor_local_path, exist_ok=True)
        if os.path.exists(f'{actor_local_path}/metrics.json'):
            with open(f'{actor_local_path}/metrics.json', 'r') as f:
                metrics = json.load(f)
            best_score = metrics['best_avg_score']
        else:
            print('Find no current best saved. Best score is set to -inf')
            best_score = -float('inf')

        cur_score = val_metrics['avg_score']

        if cur_score > best_score:
            print(f'Saving best checkpoint with score {cur_score} at {actor_local_path}')
            best_score = cur_score
            self.actor_rollout_wg.save_checkpoint_hf(actor_local_path)
            with open(f'{actor_local_path}/metrics.json', 'w') as f:
                f.write(json.dumps({'best_avg_score': best_score, 'global_step': self.global_steps})+'\n')

def compute_data_metrics_ours(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    from verl.trainer.ppo.ray_trainer import _compute_response_info
    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    # compute on/off policy stats
    off_policy_mask = batch.batch['prefix_mask'].any(-1) # [bsz, ]
    on_policy_mask = ~off_policy_mask
    off_response_length = response_length[off_policy_mask]
    on_response_length = response_length[on_policy_mask]

    off_count = off_policy_mask.sum().item()
    on_count = on_policy_mask.sum().item()
    # Without on-policy samples the ratio is not finite; avoid changing nonempty statistics with an epsilon.
    off_on_example_ratio = (off_count / on_count if on_count else
                            (float('inf') if off_count else float('nan')))

    off_sequence_score = sequence_score[off_policy_mask]
    on_sequence_score = sequence_score[on_policy_mask]

    # on/off prompt score
    # batch_size = batch.batch.batch_size[0] / n_samples
    # on_prompt_score, off_prompt_score = [], []
    # sequence_score = sequence_score.reshape(batch_size, n_samples, sequence_score.shape[-1]) # [bsz, n, l]
    # for i in range(batch_size):
    #     on_prompt_score.append(sequence_score[i][on_policy_mask[i]].mean())
    #     off_prompt_score.append(sequence_score[i][off_policy_mask[i]].mean())

    # on_prompt_score = torch.cat(on_prompt_score, dim=0)
    # off_prompt_score = torch.cat(off_prompt_score, dim=0)

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Empty-set statistics are undefined: report NaN and preserve mean/max/min for nonempty sets.
    metrics = {}
    statistic_tensors = {
        'critic/score': sequence_score,
        'critic/rewards': sequence_reward,
        'critic/advantages': valid_adv,
        'critic/returns': valid_returns,
        'response_length': response_length,
        'prompt_length': prompt_length,
    }
    if use_critic:
        statistic_tensors['critic/values'] = valid_values
        metrics['critic/vf_explained_var'] = (1.0 - return_diff_var / (return_var + 1e-5)).detach().item()
    for name, tensor in statistic_tensors.items():
        for statistic, reduce_fn in (('mean', torch.mean), ('max', torch.max), ('min', torch.min)):
            metrics[f'{name}/{statistic}'] = (reduce_fn(tensor).detach().item()
                                             if tensor.numel() else float('nan'))

    metrics.update({
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        'on_off_metrics/on_response_length_mean':
            torch.mean(on_response_length).detach().item(),
        'on_off_metrics/off_response_length_mean':
            torch.mean(off_response_length).detach().item(),
        'on_off_metrics/on_score':
            torch.mean(on_sequence_score).detach().item(),
        'on_off_metrics/off_score':
            torch.mean(off_sequence_score).detach().item(),
        'on_off_metrics/off_on_example_ratio': off_on_example_ratio,
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    })
    return metrics