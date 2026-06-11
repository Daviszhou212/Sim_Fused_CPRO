# Actor Distribution Entry Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Gaussian-actor simulation entry files choose legacy or squashed policy distributions while keeping `squashed` as the default and leaving DK-only entries unchanged.

**Architecture:** Add canonical distribution helpers and policy factories in `model.py`. Route SLDAC, SLDAC_Pathwise, and Fused-CPRO actor construction through those factories, and persist the resolved distribution in checkpoints so frozen old policies keep their own log-prob semantics.

**Tech Stack:** Python, PyTorch, `unittest`, existing PowerShell/conda `torch` test workflow.

---

### Task 1: Lock Configuration Behavior With Tests

**Files:**
- Create: `Squash/MIMO/test_actor_distribution_config.py`

- [x] **Step 1: Write failing tests**

Add tests that import entry configs, assert Gaussian actor entries default to `squashed`, assert DK entries do not expose `actor_distribution`, assert the factory can build legacy and squashed actors, and assert legacy and squashed log-probs differ for the same action.

- [ ] **Step 2: Run test to verify it fails**

Run: `& "<conda-base>\envs\torch\python.exe" -m unittest MIMO.test_actor_distribution_config`

Expected: FAIL because factory helpers and entry config fields do not exist yet.

### Task 2: Add Policy Factory and Legacy Actors

**Files:**
- Modify: `Squash/MIMO/model.py`
- Modify: `Squash/CLQR/model.py`

- [ ] **Step 1: Implement canonical distribution helpers**

Add `SQUASHED_ACTOR_DISTRIBUTION`, `LEGACY_ACTOR_DISTRIBUTION`, `normalize_actor_distribution()`, `get_action_transform_metadata(actor_distribution)`, and `build_gaussian_policy()`.

- [ ] **Step 2: Implement legacy actors**

Add legacy Gaussian policy classes that sample and evaluate a direct Normal in action space. MIMO legacy uses bounded mean for power dimensions and positive mean for regularization; CLQR legacy uses the historical direct Gaussian mean.

- [ ] **Step 3: Run focused factory tests**

Run the new test and existing squashed Gaussian tests.

### Task 3: Route Algorithms Through the Factory

**Files:**
- Modify: `Squash/MIMO/SLDAC.py`
- Modify: `Squash/CLQR/SLDAC.py`
- Modify: `Squash/MIMO/SLDAC_Pathwise.py`
- Modify: `Squash/CLQR/SLDAC_Pathwise.py`
- Modify: `Squash/MIMO/Fused_CPRO.py`
- Modify: `Squash/CLQR/Fused_CPRO.py`

- [ ] **Step 1: Use `build_gaussian_policy()` for new actors**

Resolve `args.actor_distribution`, pass it into actor construction, and keep default fallback as squashed.

- [ ] **Step 2: Save checkpoint metadata from the resolved actor**

Write `actor_distribution` and matching `action_transform` from the actor instance, not from a global constant.

- [ ] **Step 3: Load frozen policies with their checkpoint distribution**

Treat missing checkpoint distribution as legacy, instantiate old policies with that distribution, and leave mixture `logsumexp` logic unchanged.

### Task 4: Add Entry Configuration

**Files:**
- Modify Gaussian actor run entries under `Squash/MIMO/` and `Squash/CLQR/`

- [ ] **Step 1: Add `DEFAULT_ACTOR_DISTRIBUTION = "squashed"`**

Add the field to SLDAC, SLDAC_Pathwise, Fused-CPRO, CosRho, RhoNew, HRL, and PRCRL entries.

- [ ] **Step 2: Include it in `build_python_config()` and parser**

Follow the existing Python-config-priority pattern. DK-only entries remain unchanged.

### Task 5: Verify Without Writing Experiment Artifacts

**Files:**
- Test only

- [ ] **Step 1: Run py_compile**

Compile modified Python modules and entries only.

- [ ] **Step 2: Run focused unit tests**

Run new actor config tests, squashed Gaussian tests, SLDAC pathwise tests, and Fused rho bounds tests. Do not run training, plotting, or export scripts.
