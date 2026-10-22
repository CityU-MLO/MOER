import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F

def compute_sft_pure_loss(log_prob, eos_mask):
    sft_losses = -log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return sft_loss

def compute_chord_loss(log_prob, eos_mask):
    prob = torch.exp(log_prob).detach()
    phi = prob * (1 - prob)
    sft_losses = -log_prob
    sft_loss = verl_F.masked_mean(phi * sft_losses, eos_mask)
    return sft_loss

def compute_grpo_outcome_advantage_split(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   on_policy_mask: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   use_std: bool = True):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            # only include on-policy samples for mean and std calculation
            if on_policy_mask[i].item() is True:
                id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # process std
        for idx in id2std:
            if id2std[idx].item() == 0:
                id2std[idx] = torch.tensor(1.0)
        for i in range(bsz):
            if use_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = (scores[i] - id2mean[index[i]])
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

def compute_token_on_off_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    eos_mask,
    cliprange,
    clip_upper_bound,
    prefix_mask,
    off_cliprange,
    off_normalize=False,
    off_abs_cliprange=None,
    off_max_clip=None,
    off_min_clip=None,
    all_max_clip=None,
    off_policy_reshape="no_reshape", 
    off_policy_reshape_weight=1.0,
    off_policy_reshape_pow_exp=0.5,
    off_policy_scaled_tr_coeff=0.2,
    on_policy_reshape="no_reshape", 
    on_policy_reshape_weight=1.0,
    on_policy_reshape_pow_exp=0.5,
    target_probs=None,
    loss_remove_token_mean=False,
    loss_remove_clip=False,
    clip_ratio_c=3.0,
    expand_prefix_mask_to_sample=True,
    entropy=None,
    entropy_clip_ratio=0.2,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        prefix_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_c: (float)
            The lower-bound clip ratio used in Dual-Clip PPO (see https://arxiv.org/abs/1912.09729).
            Only applied to on-policy tokens with negative advantage. Must be > 1.0.
            Default: 3.0.

    Numerical semantics:
        clip_upper_bound directly sets the shared importance-ratio upper bound.
        The lower bound remains 1 - cliprange.
        entropy_clip ranks valid off-policy tokens across the full batch using
        the current prefix_mask. Sample-level expansion includes completion
        tokens of guided samples. The selected_off_mask gates both off-policy
        loss statistics and the final optimized loss. Unselected off-policy
        tokens do not become on-policy tokens. Final aggregation keeps the
        original response mask and normalization denominator.

    Returns:
        A dictionary containing pg_loss, on/off components, and monitoring metrics.

    """
    # off-policy loss
    # compute off-policy probability

    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    if on_policy_reshape == "no_reshape":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
    elif on_policy_reshape == "logp":
        ratio = log_prob - old_log_prob
    elif on_policy_reshape == "p_logp":
        ratio = torch.exp(negative_approx_kl) + on_policy_reshape_weight * negative_approx_kl
    elif on_policy_reshape == "square_root":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.sqrt(ratio)
    elif on_policy_reshape == "pow":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.pow(ratio, on_policy_reshape_pow_exp)
    elif on_policy_reshape == "p_div_p_0.1":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.1)
        f_old_prob = old_prob / (old_prob + 0.1)
        ratio = f_prob / f_old_prob
    elif on_policy_reshape == "p_div_p_0.5":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.5)
        f_old_prob = old_prob / (old_prob + 0.5)
        ratio = f_prob / f_old_prob
    else:
        raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

    on_pg_losses = -advantages * ratio
    upper_bound = clip_upper_bound
    if not loss_remove_clip and (upper_bound is None or upper_bound < 1.0 - cliprange):
        raise ValueError('clip_upper_bound must be at least 1 - cliprange')

    # Build token masks for separate probability monitoring:
    # - tokens belonging to a sample that contains a prefix (prefix_mask.any(-1)==True)
    #   AND that are NOT inside the prefix region (~prefix_mask) -> "with-prefix on-policy" token
    # - tokens belonging to a sample without any prefix -> "pure on-policy" token
    has_prefix_seq = prefix_mask.any(dim=-1, keepdim=True)  # [bsz, 1]
    with_prefix_on_mask = has_prefix_seq.expand_as(prefix_mask) & (~prefix_mask)  # [bsz, l]
    pure_on_mask = (~has_prefix_seq).expand_as(prefix_mask)  # [bsz, l]

    old_prefix_mask = prefix_mask.clone()
    # If expand_prefix_mask_to_sample is True, the prefix_mask covers all tokens that belong to
    # any sample containing a prefix (so off-policy loss is applied on the full mixed sample).
    # Otherwise, the prefix_mask only covers tokens that actually lie inside the prefix region.
    if expand_prefix_mask_to_sample:
        prefix_mask = has_prefix_seq.expand_as(prefix_mask).clone()
    else:
        prefix_mask = prefix_mask.clone()

    if loss_remove_clip is False:
        # All on-policy tokens use the same explicit upper bound.
        ratio_clipped = torch.clamp(ratio, min=1.0 - cliprange, max=upper_bound)
        on_pg_losses2 = -advantages * ratio_clipped
        on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)

        # Dual-Clip PPO lower bound (https://arxiv.org/abs/1912.09729):
        # For tokens with negative advantage, cap the loss from above by -clip_ratio_c * A,
        # i.e. final loss = min(max(L1, L2), -clip_ratio_c * A) when A < 0.
        # This prevents extremely large policy updates on negative-advantage tokens
        # whose importance ratio explodes.
        if clip_ratio_c is not None and clip_ratio_c > 1.0:
            assert clip_ratio_c > 1.0, (
                f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
                f" but got {clip_ratio_c}"
            )
            neg_adv_mask = (advantages < 0).float()
            on_pg_losses3 = -advantages * float(clip_ratio_c)
            # Measure hits before dual clipping; normalize over all valid on-policy tokens.
            on_pg_lower_clipfrac = verl_F.masked_mean(
                torch.lt(on_pg_losses3, on_pg_losses).float() * neg_adv_mask,
                (~prefix_mask) * eos_mask,
            )
            # Preserve the loss calculation order and numerical semantics.
            on_pg_losses_dual = torch.min(on_pg_losses, on_pg_losses3)
            on_pg_losses = on_pg_losses_dual * neg_adv_mask + on_pg_losses * (1.0 - neg_adv_mask)
        else:
            on_pg_lower_clipfrac = torch.tensor(0.0)

        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
    else:
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
        on_pg_clipfrac = torch.tensor(0.0)
        on_pg_lower_clipfrac = torch.tensor(0.0)

    # compute off-policy loss
    if target_probs is None:
        off_ratio = torch.exp(log_prob) # [bsz, l]
        if off_policy_reshape == "no_reshape":
            pass
        elif off_policy_reshape == "logp":
            off_ratio = log_prob * off_policy_reshape_weight
        elif off_policy_reshape == "p_logp":
            off_ratio = log_prob * off_policy_reshape_weight + off_ratio
        elif off_policy_reshape == "square_root":
            off_ratio = torch.sqrt(off_ratio)
        elif off_policy_reshape == "p_div_p_0.1":
            off_ratio = off_ratio / (off_ratio + 0.1)
        elif off_policy_reshape == "p_div_p_0.2":
            off_ratio = off_ratio / (off_ratio + 0.2)
        elif off_policy_reshape == "p_div_p_0.5":
            off_ratio = off_ratio / (off_ratio + 0.5)
        elif off_policy_reshape == "p_div_p_0.3":
            off_ratio = off_ratio / (off_ratio + 0.3)
        elif off_policy_reshape == "pow":
            off_ratio = torch.pow(off_ratio, off_policy_reshape_pow_exp)
        elif off_policy_reshape == "trust_region":
            off_ratio = off_ratio / torch.clamp(off_ratio.detach(), min=0.1)
        elif off_policy_reshape == "scaled_tr":
            off_ratio = off_ratio * (off_policy_scaled_tr_coeff / torch.clamp(off_ratio.detach(), min=0.1))
        elif off_policy_reshape == "phi":
            off_ratio = off_ratio * (1-off_ratio.detach())
        elif off_policy_reshape == "entropy_clip":
            pass
        else:
            raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
    else:
        assert target_probs.shape == log_prob.shape
        off_ratio = torch.exp(log_prob) / (target_probs+1e-6)
        # off_ratio[log_prob == 0] = 0
        off_ratio = off_ratio * prefix_mask
        # assert ((target_probs > 0) == prefix_mask).all()

    # clip off-policy ratio
    if off_max_clip is not None:
        off_ratio = torch.clamp(off_ratio, max=off_max_clip)
        off_ratio_max_clip_frac = verl_F.masked_mean((off_ratio == off_max_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_max_clip_frac = torch.tensor(0.0)

    if off_min_clip is not None:
        off_ratio = torch.clamp(off_ratio, min=off_min_clip)
        off_ratio_min_clip_frac = verl_F.masked_mean((off_ratio == off_min_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_min_clip_frac = torch.tensor(0.0)

    # Without entropy filtering, all off-policy tokens remain selected.
    entropy_mask = torch.ones_like(prefix_mask, dtype=torch.bool)
    if off_policy_reshape == "entropy_clip":
        if not 0.0 <= float(entropy_clip_ratio) <= 1.0:
            raise ValueError('entropy_clip_ratio must be between 0 and 1')
        # Select the global top-k valid tokens under the current prefix_mask; other positions are False.
        # Sample-level expansion includes non-prefix tokens in mixed samples.
        assert entropy is not None, (
            "entropy must be provided when off_policy_reshape == 'entropy_clip'"
        )
        # Keep the current, potentially expanded prefix_mask to preserve the selection algorithm.
        valid_prefix_mask = (prefix_mask.bool() & eos_mask.bool())
        # Replace invalid positions with -inf so they are never picked by top-k.
        masked_entropy = torch.where(
            valid_prefix_mask,
            entropy.detach(),
            torch.full_like(entropy, float("-inf")),
        )
        num_valid = int(valid_prefix_mask.sum().item())
        k = int(num_valid * float(entropy_clip_ratio))
        entropy_mask = torch.zeros_like(prefix_mask, dtype=torch.bool)
        if k > 0:
            flat_entropy = masked_entropy.reshape(-1)
            topk_idx = torch.topk(flat_entropy, k=k, largest=True).indices
            flat_entropy_mask = entropy_mask.reshape(-1)
            flat_entropy_mask[topk_idx] = True
            entropy_mask = flat_entropy_mask.reshape_as(prefix_mask)

    prefix_mask = prefix_mask.float()
    old_prefix_mask = old_prefix_mask.float()
    selected_off_mask = prefix_mask * entropy_mask.float()
    selected_eos_mask = selected_off_mask * eos_mask
    off_ratio_mean = verl_F.masked_mean(off_ratio, selected_eos_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = log_prob.new_zeros(())

    # Apply selection to the differentiable token losses, not just the metrics.
    off_pg_losses = -advantages * off_ratio * selected_off_mask
    off_pg_loss = verl_F.masked_mean(off_pg_losses, selected_eos_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = log_prob.new_zeros(())
    off_pg_clipfrac = torch.tensor(0.0)

    # Keep the original on-policy mask: rejected prefix tokens contribute neither
    # off-policy nor on-policy loss. The final eos normalization stays unchanged.
    pg_losses = off_pg_losses + on_pg_losses * (1 - prefix_mask)

    # log on/off probs
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = verl_F.masked_mean(off_policy_probs, old_prefix_mask * eos_mask)
    if off_policy_prob.isnan().item() is True:
        off_policy_prob = torch.tensor(0.0)
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = verl_F.masked_mean(on_policy_probs, (1.0-old_prefix_mask) * eos_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0)

    # log probs separately for "with-prefix on-policy" tokens and "pure on-policy" tokens
    with_prefix_on_mask_f = with_prefix_on_mask.float()
    pure_on_mask_f = pure_on_mask.float()
    on_policy_prob_with_prefix = verl_F.masked_mean(
        on_policy_probs, with_prefix_on_mask_f * eos_mask
    )
    if on_policy_prob_with_prefix.isnan().item() is True:
        on_policy_prob_with_prefix = torch.tensor(0.0)
    on_policy_prob_pure = verl_F.masked_mean(
        on_policy_probs, pure_on_mask_f * eos_mask
    )
    if on_policy_prob_pure.isnan().item() is True:
        on_policy_prob_pure = torch.tensor(0.0)

    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        eos_mask = eos_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask

    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        # print(f'no token mean: mean normalization {eos_mask.shape[-1]}')
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

    return {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_lower_clipfrac": on_pg_lower_clipfrac,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,
        "on_policy_prob_with_prefix": on_policy_prob_with_prefix,
        "on_policy_prob_pure": on_policy_prob_pure,
        "off_ratio_mean": off_ratio_mean,
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    }