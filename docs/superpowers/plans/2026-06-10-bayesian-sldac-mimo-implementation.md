# Bayesian Critic SLDAC MIMO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `Bayesian_SLDAC_MIMO/`, run a 100-episode `b100_q1` comparison between legacy SLDAC and Bayesian critic SLDAC, and generate comparison plots.

**Architecture:** Add an isolated package that copies the `SLDAC_code/MIMO1` runtime semantics: seed 0, CPU, legacy Gaussian actor, MIMO lower projection, SLDAC time scales, and fixed run tags. The package supplies both a legacy single-critic baseline and a Bayesian ensemble critic variant so the final comparison is generated from the same isolated runner and artifact layout.

**Tech Stack:** Python, NumPy, PyTorch, SciPy, Matplotlib, `scipy.io.savemat/loadmat`, `unittest`, conda env `torch_work`.

---

### Task 1: Package Skeleton And Configuration

**Files:**
- Create: `Bayesian_SLDAC_MIMO/__init__.py`
- Create: `Bayesian_SLDAC_MIMO/config.py`
- Create: `Bayesian_SLDAC_MIMO/artifact_paths.py`
- Create: `Bayesian_SLDAC_MIMO/tests/test_config.py`

- [ ] **Step 1: Write failing configuration and path tests**

Create tests asserting `seed=0`, `device="cpu"`, `episode=60` default, `b100_q1=(500,500,100,1)`, 100-episode override support, and all output paths remain inside `Bayesian_SLDAC_MIMO/`.

- [ ] **Step 2: Run RED**

Run:

```powershell
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m unittest Bayesian_SLDAC_MIMO.tests.test_config
```

Expected: import failure because the package does not exist yet.

- [ ] **Step 3: Implement configuration and path helpers**

Define `DEFAULT_CONFIG`, `SLDAC_RUNS`, `make_run_config(tag, episode_override=None)`, `ensure_dir(path)`, and `build_compare_output_dir(label)`.

- [ ] **Step 4: Run GREEN**

Run the same unittest command and expect PASS.

### Task 2: Legacy MIMO Environment And Actor Semantics

**Files:**
- Create: `Bayesian_SLDAC_MIMO/environment.py`
- Create: `Bayesian_SLDAC_MIMO/model.py`
- Create: `Bayesian_SLDAC_MIMO/tests/test_environment_random_stream.py`
- Create: `Bayesian_SLDAC_MIMO/tests/test_model_legacy_action.py`

- [ ] **Step 1: Write failing tests**

Tests must verify `Environment_MIMO(seed=0, Nt=8, UE_num=4)` reset/step shape, lower projection of `action <= 0` to `1e-6`, no upper clipping for positive action, actor mean in `(0, 2.5)`, plain Gaussian log-prob shape, and possibility of unbounded samples.

- [ ] **Step 2: Run RED**

Run:

```powershell
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m unittest Bayesian_SLDAC_MIMO.tests.test_environment_random_stream Bayesian_SLDAC_MIMO.tests.test_model_legacy_action
```

Expected: import failures.

- [ ] **Step 3: Implement environment and model**

Copy the runtime semantics from `SLDAC_code/MIMO1/environment.py` and `SLDAC_code/MIMO1/model.py` into the isolated package. Keep the legacy actor as `GaussianPolicy_MIMO`, the critic network as `CriticNetMIMO`, and helper functions for flatten/writeback.

- [ ] **Step 4: Run GREEN**

Run the same unittest command and expect PASS.

### Task 3: Buffer, Legacy Critic, Bayesian Critic, And Risk Correction

**Files:**
- Create: `Bayesian_SLDAC_MIMO/buffer.py`
- Create: `Bayesian_SLDAC_MIMO/bayesian_critic.py`
- Create: `Bayesian_SLDAC_MIMO/tests/test_bayesian_critic.py`

- [ ] **Step 1: Write failing critic tests**

Tests must cover buffer window shape, single-critic output shape, ensemble `mean/std` shape, finite outputs, objective risk correction subtracting std, constraint risk correction adding std, and `beta_uncertainty=0` returning ensemble mean.

- [ ] **Step 2: Run RED**

Run:

```powershell
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m unittest Bayesian_SLDAC_MIMO.tests.test_bayesian_critic
```

Expected: import failures.

- [ ] **Step 3: Implement buffer and critics**

Implement `DataStorage`, `LegacyCritic`, `BayesianCritic`, `risk_correct_q_values`, and `normalize_q_hat_like_sldac_code`. `LegacyCritic` is ensemble size 1 behavior; `BayesianCritic` owns independent member heads and target heads.

- [ ] **Step 4: Run GREEN**

Run the same unittest command and expect PASS.

### Task 4: Lagrangian CSSCA Solver

**Files:**
- Create: `Bayesian_SLDAC_MIMO/lagrangian_cssca.py`
- Create: `Bayesian_SLDAC_MIMO/tests/test_lagrangian_cssca.py`

- [ ] **Step 1: Write failing solver tests**

Tests must cover objective-only descent, feasible constrained update, infeasible/solver-failure conservative fallback to `theta_t`, finite `theta_bar`, `cssca_status`, and no CVX/MOSEK import on the default path.

- [ ] **Step 2: Run RED**

Run:

```powershell
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m unittest Bayesian_SLDAC_MIMO.tests.test_lagrangian_cssca
```

Expected: import failure.

- [ ] **Step 3: Implement solver**

Use `scipy.optimize.minimize` with SLSQP on the primal surrogate problem. Return `CsscaResult(theta_bar, status, success, step_norm, message)`. If optimization fails or produces non-finite values, return `theta_t`.

- [ ] **Step 4: Run GREEN**

Run the same unittest command and expect PASS.

### Task 5: SLDAC Loop, Runner, And Plotting

**Files:**
- Create: `Bayesian_SLDAC_MIMO/sldac.py`
- Create: `Bayesian_SLDAC_MIMO/run_bayesian_sldac_mimo.py`
- Create: `Bayesian_SLDAC_MIMO/plot_compare.py`
- Create: `Bayesian_SLDAC_MIMO/tests/test_sldac_smoke.py`

- [ ] **Step 1: Write failing smoke and artifact tests**

Tests must verify small isolated runs write only under `Bayesian_SLDAC_MIMO/Trash/`, return reward/cost arrays, and produce diagnostics containing `objective_avg`, `constraint_violation`, `cssca_status`, `q_std_objective`, and `ensemble_size`.

- [ ] **Step 2: Run RED**

Run:

```powershell
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m unittest Bayesian_SLDAC_MIMO.tests.test_sldac_smoke
```

Expected: import failure.

- [ ] **Step 3: Implement loop, runner, and plotting**

Implement `SLDAC_main(args, example_name, mode)` where `mode` is `legacy` or `bayesian`. The runner must support `--run-tag b100_q1`, `--episode 100`, `--mode compare`, and `--output-root <dir>`. Plotting must write `compare_b100_q1_100ep.png` and `.pdf`.

- [ ] **Step 4: Run GREEN**

Run smoke unittest and expect PASS.

### Task 6: Verification And 100-Episode Comparison

**Files:**
- Runtime outputs under `Bayesian_SLDAC_MIMO/outputs/compare_b100_q1_100ep_<timestamp>/`
- Optional moved smoke artifacts under `Bayesian_SLDAC_MIMO/Trash/`

- [ ] **Step 1: Run compile and all unit tests**

Run:

```powershell
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m compileall Bayesian_SLDAC_MIMO
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m unittest discover Bayesian_SLDAC_MIMO/tests
```

Expected: PASS.

- [ ] **Step 2: Run isolated 100-episode comparison**

Run:

```powershell
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" Bayesian_SLDAC_MIMO\run_bayesian_sldac_mimo.py --mode compare --run-tag b100_q1 --episode 100
```

Expected: a new output directory under `Bayesian_SLDAC_MIMO/outputs/compare_b100_q1_100ep_<timestamp>/` containing legacy and Bayesian `.mat`, diagnostics, logs, and comparison plots.

- [ ] **Step 3: Inspect and summarize results**

Load the result summary JSON and report final objective/cost averages, constraint curves, and plot paths.
