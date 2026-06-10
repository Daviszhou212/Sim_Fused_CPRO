# Spantest

`Spantest` is a self-contained MIMO1 span-CSSCA experiment directory. It does not
run `SLDAC_code/MIMO1/MIMO_main.py` and does not import the old `SLDAC_code`
entrypoint.

## What It Compares

- `full`: original full-parameter CVXPY CSSCA actor subproblem.
- `span`: exact full-gradient span reduction.
- `active`: active-gradient span reduction.

The actor remains the full MIMO1 Gaussian policy. Only the CVXPY decision space
inside the CSSCA actor update is changed.

## All-in-One Entry

Use `Spantest/main.py` when you want one command to run the configured
comparison, write `csv/mat/json`, and generate `png/pdf` figures:

```powershell
& "D:\anaconda3\envs\torch\python.exe" Spantest\main.py
```

For a fast pipeline check:

```powershell
& "D:\anaconda3\envs\torch\python.exe" Spantest\main.py --smoke --run-name entry_smoke
```

The top of `Spantest/main.py` contains the default output path, run name,
solver switches, plotting switch, and the MIMO1 experiment parameters copied
from `SLDAC_code/MIMO1/MIMO_main.py`:

- `RUN_ORIGINAL_SLDAC = False`
- `RUN_LOWER_DIMENSION_SLDAC = True`
- `RUN_ACTIVE_SLDAC = True`

- `T = 500`
- `NUM_NEW_DATA = 100`
- `WINDOW = 10000`
- `GRAD_T = T`
- `EPISODE = 60`
- `UPDATE_TIME_PER_EPISODE = 10`
- `NUM_UPDATE_TIME = EPISODE * UPDATE_TIME_PER_EPISODE`
- `Q_UPDATE_TIME = 1`
- `PRINT_INTERVAL`

Edit those values at the top of `Spantest/main.py` for routine experiments.
CLI arguments are kept minimal and are mainly for smoke checks or one-off
output naming.

By default, the original full-parameter `full` solver is skipped to save time.
Set `RUN_ORIGINAL_SLDAC = True` when you need to regenerate the original SLDAC
baseline in the same run.

Progress is printed by default. `PRINT_INTERVAL = k` means one progress line is
printed every `k` logged episodes. For example, `PRINT_INTERVAL = 5` prints at
episode 5, 10, 15, and so on. Each progress line includes the current
solver, episode, actor update, CSSCA branch/status, decision dimension, gradient
rank, CVXPY solve time, objective cost, and constraint cost.

## Separate Entries

The original separate entries are intentionally preserved. Use
`Spantest.run_experiment` when you want direct access to the experiment runner:

```powershell
& "D:\anaconda3\envs\torch\python.exe" -m Spantest.run_experiment --run-name demo --solvers full span active
```

For a faster smoke run:

```powershell
& "D:\anaconda3\envs\torch\python.exe" -m Spantest.run_experiment --run-name smoke --solvers span active --T 5 --grad-T 5 --num-new-data 5 --episode 2 --num-update-time 2 --window 20
```

Outputs are written under `Spantest/outputs/<run-name>/` and are intentionally
not ignored, so `mat/png/pdf/csv/json` artifacts can be committed when desired.

## Included Smoke Output

This directory includes a small full/span/active comparison at:

```text
Spantest/outputs/smoke_full_span_active/
```

It uses a short MIMO1 run only to verify the full pipeline. The actor parameter
dimension is `25994` in all three solvers. In the included smoke timing:

- `full`: CVXPY decision dimension `25994`, solve time `2.665132s`.
- `span`: CVXPY decision dimension `5`, solve time `0.0403922s`.
- `active`: CVXPY decision dimension `3`, solve time `0.035898s`.

The smoke output includes:

- `curves.mat`
- `summary.csv`
- `timing.csv`
- `metadata.json`
- objective/cost/timing/dimension figures in both `png` and `pdf`.
