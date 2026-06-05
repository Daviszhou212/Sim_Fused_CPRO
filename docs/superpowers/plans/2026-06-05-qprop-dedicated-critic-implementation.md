# Dedicated Q-Prop Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the dedicated Q-Prop critic described in `docs/superpowers/specs/2026-06-05-qprop-dedicated-critic-design.md` without creating a new branch.

**Architecture:** Add a sidecar `QPropCritic` in `MIMO3` and `CLQR2`. Keep the SLDAC main critic as the source of score signal, and use the sidecar critic only for Q-Prop control variate and pathwise control. The sidecar critic uses average-cost TD target `cost - func_value + next_q`, with target actor actions detached during critic training.

**Tech Stack:** Python, PyTorch, `unittest`, PowerShell 5.1, conda env `torch_work` for this C-drive workspace.

---

### Task 1: RED tests for QPropCritic modules

**Files:**
- Create: `MIMO3/test_qprop_critic.py`
- Create: `CLQR2/test_qprop_critic.py`

- [ ] **Step 1: Add tests that import `QPropCritic`**

The tests must assert head count, finite update diagnostics, differentiable target head value, detached actor target action during TD update, and direct use of `costs_batch[:, 0]`.

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location "MIMO3"
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m unittest "test_qprop_critic.py"
Pop-Location

Push-Location "CLQR2"
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m unittest "test_qprop_critic.py"
Pop-Location
```

Expected: FAIL because `qprop_critic.py` does not exist.

### Task 2: Implement QPropCritic modules

**Files:**
- Create: `MIMO3/qprop_critic.py`
- Create: `CLQR2/qprop_critic.py`

- [ ] **Step 1: Implement model selection and head validation**

`head_count = 1 + constraint_dim`. MIMO requires `constraint_dim == 4`; CLQR requires `constraint_dim == 1`; otherwise raise `ValueError`.

- [ ] **Step 2: Implement average-cost TD update**

`update_from_replay(...)` samples uniformly without replacement, computes detached actor mean target action, and trains each head with:

```text
y_h = costs_batch[:, h] - func_value[h] + target_h(next_state, next_action)
```

- [ ] **Step 3: Implement diagnostics and value accessors**

Return numeric arrays for critic loss, TD error mean, target mean, prediction mean, replay batch size, control-source code, and target-action code. Expose `head_value(...)`, `all_head_values(...)`, `flatten_parameters()`, and `head_count`.

- [ ] **Step 4: Run QPropCritic tests and verify GREEN**

Run the two `test_qprop_critic.py` suites. Expected: PASS.

### Task 3: RED tests for Pathwise SLDAC dedicated control source

**Files:**
- Modify: `MIMO3/test_sldac_pathwise.py`
- Modify: `CLQR2/test_sldac_pathwise.py`

- [ ] **Step 1: Add tests for dedicated source and diagnostics**

Tests must pass a `QPropCritic` into `_compute_pathwise_gradients(...)` and assert:

- dedicated critic produces different control values than the main critic when its weights differ;
- `qprop_control_source_code == 1.0`;
- `qprop_pathwise_grad_ratio` and `qprop_score_grad_ratio` are finite and in `[0, 1]`;
- `use_qprop_dedicated_critic=False` keeps v1 main-critic source code `0.0`.

- [ ] **Step 2: Run focused tests and verify RED**

Expected: FAIL because `_compute_pathwise_gradients(...)` does not accept the dedicated critic fields yet.

### Task 4: Wire QPropCritic into SLDAC_Pathwise

**Files:**
- Modify: `MIMO3/SLDAC_Pathwise.py`
- Modify: `CLQR2/SLDAC_Pathwise.py`
- Modify: `MIMO3/run_mimo_sldac_pathwise.py`
- Modify: `CLQR2/run_clqr_sldac_pathwise.py`

- [ ] **Step 1: Add config fields**

Add dedicated critic config fields to checkpoint/config snapshots and run defaults:

```text
use_qprop_dedicated_critic
qprop_critic_update_steps
qprop_replay_batch_size
qprop_target_action_mode
qprop_critic_lr_scale
qprop_target_tau_reward
qprop_target_tau_cost
```

- [ ] **Step 2: Add diagnostics fields**

Add numeric diagnostics for critic losses, TD means, replay batch size, source code, target-action code, and gradient norm ratios.

- [ ] **Step 3: Extend gradient helpers**

Let `_compute_qprop_conservative_gradient(...)` receive both the score critic and an optional control critic. Score signal always comes from the main critic; control signal and pathwise loss use the dedicated critic when enabled.

- [ ] **Step 4: Update main loop**

Instantiate `QPropCritic` only for `qprop_conservative` when enabled. Before actor-gradient extraction on completed Q update cycles, update the sidecar critic from replay buffers with detached target actor actions and explicit tau values.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run both `test_sldac_pathwise.py` suites. Expected: PASS.

### Task 5: Final verification

**Files:**
- Read-only verification.

- [ ] **Step 1: Compile changed modules**

```powershell
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m py_compile `
  "MIMO3\qprop_critic.py" `
  "CLQR2\qprop_critic.py" `
  "MIMO3\SLDAC_Pathwise.py" `
  "CLQR2\SLDAC_Pathwise.py" `
  "MIMO3\run_mimo_sldac_pathwise.py" `
  "CLQR2\run_clqr_sldac_pathwise.py"
```

- [ ] **Step 2: Run focused unittest suites**

```powershell
Push-Location "MIMO3"
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m unittest "test_qprop_critic.py" "test_sldac_pathwise.py"
Pop-Location

Push-Location "CLQR2"
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m unittest "test_qprop_critic.py" "test_sldac_pathwise.py"
Pop-Location
```

- [ ] **Step 3: Inspect worktree**

Run `git status --short` and confirm no formal experiment artifacts were created. Temporary simulation is not required for this implementation pass unless explicitly requested after tests pass.
