# Q-Prop Pathwise SLDAC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative Q-Prop policy-gradient mode to `SLDAC_Pathwise` for both MIMO3 and CLQR2.

**Architecture:** Keep the SLDAC update and critic TD update unchanged. Add a new actor-gradient mode inside `SLDAC_Pathwise.py`, dispatching beside existing `stochastic_pathwise` and `deterministic_dpg` modes. Reuse target critic heads for both score residual and pathwise control terms.

**Tech Stack:** Python, PyTorch, `unittest`, conda environment `torch_work` for this C-drive workspace.

---

### Task 1: RED tests for Q-Prop mode

**Files:**
- Modify: `MIMO3/test_sldac_pathwise.py`
- Modify: `CLQR2/test_sldac_pathwise.py`

- [ ] **Step 1: Add tests that require the new mode and helpers**

Add imports for:

```python
POLICY_GRADIENT_MODES,
_compute_conservative_qprop_eta,
```

Add tests that assert:

```python
self.assertIn("qprop_conservative", POLICY_GRADIENT_MODES)
```

and verify `_compute_pathwise_gradients(...)` accepts `action_batch_torch`,
returns finite gradients for `qprop_conservative`, includes diagnostics fields
`qprop_eta`, `qprop_covariance`, `score_grad_norm`, `pathwise_grad_norm`,
`combined_grad_norm`, and leaves target critic `.grad` as `None`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
Push-Location "MIMO3"
& "<conda_base>\envs\torch_work\python.exe" -m unittest "test_sldac_pathwise.py"
Pop-Location

Push-Location "CLQR2"
& "<conda_base>\envs\torch_work\python.exe" -m unittest "test_sldac_pathwise.py"
Pop-Location
```

Expected: FAIL because `qprop_conservative` and Q-Prop helpers do not exist yet.

### Task 2: GREEN implementation in `MIMO3`

**Files:**
- Modify: `MIMO3/SLDAC_Pathwise.py`

- [ ] **Step 1: Add Q-Prop mode and diagnostics fields**

Extend `POLICY_GRADIENT_MODES` and diagnostics dictionaries with the new Q-Prop fields.

- [ ] **Step 2: Add helper functions**

Implement:

```python
_preprocess_score_signal(example_name, q_values_torch, head_idx)
_compute_taylor_control_signal(critic, state_batch_torch, action_batch_torch, mu_torch, head_idx)
_compute_conservative_qprop_eta(score_signal, control_signal)
_compute_qprop_conservative_gradient(...)
```

- [ ] **Step 3: Extend `_compute_pathwise_gradients`**

Add `action_batch_torch` to the signature and dispatch `policy_gradient_mode == "qprop_conservative"` to the new helper.

- [ ] **Step 4: Update call sites**

Pass `action_batch_torch` from `SLDAC_Pathwise_main()`.

### Task 3: GREEN implementation in `CLQR2`

**Files:**
- Modify: `CLQR2/SLDAC_Pathwise.py`

- [ ] **Step 1: Mirror the MIMO3 implementation**

Apply the same Q-Prop mode, helper functions, diagnostics fields, signature change, and call-site change.

### Task 4: Verification

**Files:**
- Read-only verification of changed Python files.

- [ ] **Step 1: Compile changed modules**

Run:

```powershell
& "<conda_base>\envs\torch_work\python.exe" -m py_compile `
  "MIMO3\SLDAC_Pathwise.py" `
  "CLQR2\SLDAC_Pathwise.py" `
  "MIMO3\run_mimo_sldac_pathwise.py" `
  "CLQR2\run_clqr_sldac_pathwise.py"
```

Expected: PASS.

- [ ] **Step 2: Run focused unit tests**

Run the two `test_sldac_pathwise.py` suites. Expected: PASS.

- [ ] **Step 3: Confirm no training artifacts were generated**

Run `git status --short` and confirm no new experiment outputs/checkpoints were created by this implementation pass.
