# Squashed Gaussian Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the squashed Gaussian actor contract from `docs/superpowers/specs/2026-06-08-squashed-gaussian-policy-design.md` for SLDAC, Pathwise, and Fused-CPRO.

**Architecture:** Keep actor distribution math inside `model.py`, keep SLDAC / Pathwise / Fused-CPRO training loops separate, and adapt only Fused-CPRO component density helpers. The Fused-CPRO mixture formula, rho semantics, lower bounds, labels, and offline/online mixing remain unchanged.

**Tech Stack:** Python, PyTorch, `unittest`, PowerShell on Windows. No training, plotting, export, or fixed-output experiment scripts.

---

## Reduced Test Policy

The user requested less unnecessary TDD ceremony. Do not create a red/green test for every edit. Keep only focused checks that prove the new distribution contract and Fused-CPRO invariants:

- actor support bounds and finite log-prob;
- manual `Normal(raw) - log_det` numerical equality for one MIMO and one CLQR case;
- Fused-CPRO mixture keeps `logsumexp(log rho + log pi_n)` and preserves `log_std_leaf.grad`;
- DK deterministic sample is unchanged while DK log-prob is finite at clipped boundaries;
- `py_compile` for touched modules.

## Files

- Modify: `MIMO3/model.py`
- Modify: `CLQR2/model.py`
- Modify: `MIMO3/Fused_CPRO.py`
- Modify: `CLQR2/Fused_CPRO.py`
- Modify: `dk_policies.py`
- Modify: `MIMO3/SLDAC.py`
- Modify: `CLQR2/SLDAC.py`
- Modify: `MIMO3/SLDAC_Pathwise.py`
- Modify: `CLQR2/SLDAC_Pathwise.py`
- Add/Modify focused tests under `MIMO3/` and `CLQR2/`

## Tasks

### Task 1: Actor Transform Helpers

- [x] Add `ACTION_EPS = 1e-6` and `ACTION_INVERSE_EPS = 1e-6` in both active `model.py` files, with Chinese comments.
- [x] Add helper functions for finite sigmoid interval, softplus-positive, and tanh interval transforms.
- [x] Add inverse/log-det helpers that return `-inf` for actor actions outside open support, without silently clamping actor component densities.
- [x] Keep `EPS = 0.003` only for existing initialization behavior.

### Task 2: Actor Classes

- [x] Change MIMO network `forward()` outputs to raw Gaussian loc, not action-space means.
- [x] Make `GaussianPolicy_MIMO.mean_action_tensor()` return transformed MIMO action: power in `(ACTION_EPS, 2.5)`, reg in `(ACTION_EPS, +inf)`.
- [x] Make `GaussianPolicy_MIMO.sample_action_tensor()` sample raw Gaussian and return transformed action.
- [x] Make `GaussianPolicy_MIMO.evaluate_action()` invert transformed actions and subtract the Jacobian log-det.
- [x] Keep transformed Gaussian logic on the MIMO / CLQR actors that remain in the mainline.
- [x] Make `GaussianPolicy_CLQR` use `1.5 * tanh(raw)` for mean/sample and transformed density for `evaluate_action()`.

### Task 3: Fused-CPRO Component Density

- [x] Change `_mean_action()` to call `actor.mean_action_tensor()` instead of `actor.net()`.
- [x] Keep `_log_prob_batch()` returning `(log_prob, log_std_leaf)` and make it use transformed actor density while preserving `log_std_leaf.grad`.
- [x] Make `FrozenActorPolicy.log_prob_batch()` call the frozen actor's transformed `evaluate_action()`.
- [x] Change `HeuristicGaussianPolicy.log_prob_batch()` in both Fused-CPRO files to bounded smoothing density, with internal open-support clamp only for DK density evaluation.
- [x] Do not change `_build_mixture_log_prob()` structure except component density sources.

### Task 4: Shared DK Policy

- [x] Update `dk_policies.py` `HeuristicGaussianPolicy.log_prob_batch()` to match bounded DK smoothing density.
- [x] Keep `sample_action()` returning deterministic mean without density clamp side effects.

### Task 5: Checkpoint Metadata

- [x] Add shared checkpoint metadata helper in SLDAC and Pathwise checkpoint save paths.
- [x] Include `actor_distribution = "squashed_gaussian_v1"` and transform metadata in new checkpoints.
- [x] In Fused-CPRO checkpoint loading, warn when loading a checkpoint missing this metadata.

### Task 6: Focused Verification

- [x] Add or update minimal unit tests for actor bounds, finite log-prob, support masks, manual log-prob equality, Fused mixture gradient, and DK finite boundary density.
- [x] Run `py_compile` on touched modules.
- [x] Run only focused unit tests that do not write experiment artifacts.
- [x] Inspect `git diff` to confirm algorithm files remain separate and Fused-CPRO mixture/rho logic is unchanged.
