# Multi-Cell MIMO Independent SLDAC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent `MultiCell_MIMO/` package implementing the v1 multi-cell SLDAC baseline from `docs/superpowers/specs/2026-06-10-multicell-mimo-independent-sldac-design.md`.

**Architecture:** Keep the new scenario isolated from `MIMO/`. Implement a CTDE baseline with multi-cell environment, shared local actor, centralized multi-head critic, Lagrangian-dual CSSCA solver, explicit `critic_target_mode`, isolated artifact/checkpoint roots, and focused unit tests.

**Tech Stack:** Python, NumPy, PyTorch, SciPy, optional CVXPY fallback, `unittest`.

---

## File Map

- Create `MultiCell_MIMO/__init__.py`: package marker and public version.
- Create `MultiCell_MIMO/config.py`: default top-level runtime config, CLI merge protection, enum validation.
- Create `MultiCell_MIMO/seed_utils.py`: NumPy/Torch seeding and device resolution.
- Create `MultiCell_MIMO/artifact_paths.py`: isolated output/checkpoint/trash path helpers.
- Create `MultiCell_MIMO/checkpoint.py`: checkpoint schema, CPU state dict, save-disabled behavior.
- Create `MultiCell_MIMO/buffer.py`: fixed rolling transition buffer.
- Create `MultiCell_MIMO/environment.py`: multi-cell MIMO queueing environment.
- Create `MultiCell_MIMO/model.py`: squashed shared local actor and flatten/restore helpers.
- Create `MultiCell_MIMO/critic.py`: dynamic multi-head average-cost critic with source-compatible and tex-strict TD target modes.
- Create `MultiCell_MIMO/cssca.py`: Lagrangian-dual CSSCA objective/feasible update with CVXPY fallback and fail-fast validation.
- Create `MultiCell_MIMO/sldac.py`: independent SLDAC main loop.
- Create `MultiCell_MIMO/run_multicell_sldac.py`: isolated top-config entrypoint.
- Create `MultiCell_MIMO/tests/test_config.py`
- Create `MultiCell_MIMO/tests/test_checkpoint.py`
- Create `MultiCell_MIMO/tests/test_environment.py`
- Create `MultiCell_MIMO/tests/test_model.py`
- Create `MultiCell_MIMO/tests/test_critic.py`
- Create `MultiCell_MIMO/tests/test_cssca.py`
- Create `MultiCell_MIMO/tests/test_sldac_smoke.py`

## Task 1: Core Contract Tests

**Files:**
- Create tests under `MultiCell_MIMO/tests/`

- [ ] **Step 1: Write focused failing tests**

Create tests that assert:

- config validates `critic_target_mode`, `actor_parameterization`, `log_std_mode`.
- checkpoint saving is disabled when `save_final_checkpoint=0`.
- environment reset/step shapes and per-user costs are finite.
- model action support and joint log-prob are valid.
- critic supports dynamic heads and both TD target modes.
- CSSCA Lagrangian-dual returns finite objective/feasible updates and fails fast on invalid solver output.
- SLDAC smoke runs in memory with temp roots and no formal entrypoint call.

- [ ] **Step 2: Run tests and verify RED**

Run with explicit torch conda Python:

```powershell
& "<torch-python>" -m unittest discover -s MultiCell_MIMO/tests -v
```

Expected: import failures because `MultiCell_MIMO` modules do not exist yet.

## Task 2: Runtime Scaffolding

**Files:**
- Create package support modules.

- [ ] **Step 1: Implement package/config/seed/path/checkpoint/buffer**

Implement:

- `build_default_config()`
- `merge_cli_config()`
- `validate_config()`
- `set_global_seed()`
- `resolve_torch_device()`
- isolated artifact path builders
- checkpoint save that no-ops when disabled
- fixed-size rolling buffer

- [ ] **Step 2: Run focused support tests**

```powershell
& "<torch-python>" -m unittest MultiCell_MIMO.tests.test_config MultiCell_MIMO.tests.test_checkpoint -v
```

Expected: PASS.

## Task 3: Environment And Actor

**Files:**
- Create `environment.py`
- Create `model.py`

- [ ] **Step 1: Implement multi-cell environment**

Use deterministic per-instance RNG, receiver-cell ownership for cross-cell channels, finite queues, and per-user delay costs.

- [ ] **Step 2: Implement shared local actor**

Implement local MLP, squashed action transform, inverse/log-prob, joint action sampling, local sampling, and flatten/restore with shared `log_std`.

- [ ] **Step 3: Run environment/model tests**

```powershell
& "<torch-python>" -m unittest MultiCell_MIMO.tests.test_environment MultiCell_MIMO.tests.test_model -v
```

Expected: PASS.

## Task 4: Critic And CSSCA

**Files:**
- Create `critic.py`
- Create `cssca.py`

- [ ] **Step 1: Implement critic**

Implement dynamic `1 + constraint_dim` heads, online/target critic pairs, `source_compatible` and `tex_strict` TD target modes, gamma smoothing, and `critic_value()`.

- [ ] **Step 2: Implement Lagrangian-dual CSSCA**

Port the current dual objective/feasible math into an independent module. Validate finite shapes and fail fast before actor restore.

- [ ] **Step 3: Run critic/CSSCA tests**

```powershell
& "<torch-python>" -m unittest MultiCell_MIMO.tests.test_critic MultiCell_MIMO.tests.test_cssca -v
```

Expected: PASS.

## Task 5: SLDAC Loop And Entry

**Files:**
- Create `sldac.py`
- Create `run_multicell_sldac.py`

- [ ] **Step 1: Implement `sldac.py`**

Implement in-memory SLDAC loop with objective-cost semantics, `hat_J`, critic updates, shared actor score-function gradients, CSSCA update, optional checkpoint save, and isolated roots.

- [ ] **Step 2: Implement entrypoint**

Top-level Python defaults are authoritative. CLI is optional and protected by config merge rules. Formal entrypoint writes only to `MultiCell_MIMO/outputs` and `MultiCell_MIMO/checkpoints`.

- [ ] **Step 3: Run smoke test**

```powershell
& "<torch-python>" -m unittest MultiCell_MIMO.tests.test_sldac_smoke -v
```

Expected: PASS and no artifacts outside temp roots.

## Task 6: Verification

**Files:**
- All new files.

- [ ] **Step 1: Compile**

```powershell
& "<torch-python>" -m py_compile (Get-ChildItem -Path MultiCell_MIMO -Filter *.py -Recurse | ForEach-Object FullName)
```

Expected: no syntax errors.

- [ ] **Step 2: Run full focused suite**

```powershell
& "<torch-python>" -m unittest discover -s MultiCell_MIMO/tests -v
```

Expected: PASS.

- [ ] **Step 3: Audit output paths**

Run:

```powershell
git status --short
```

Expected: new `MultiCell_MIMO/` and plan files only, plus any pre-existing unrelated `AGENTS.md` change. No `.mat`, `.png`, `.pdf`, `.pt` artifacts outside `Trash` or temp roots.
