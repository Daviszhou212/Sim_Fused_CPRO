# Final simulation result backup

This folder backs up the `.mat` files, plotting scripts, and final PNG figures
that were confirmed to be used by the current final CLQR and MIMO figures.

## Contents

- `CLQR2/outputs/...`: CLQR source `.mat` files and final comparison figures.
- `MIMO3/outputs/...`: MIMO source `.mat` files and final comparison figures.
- `CLQR2/CLQR_main.py`: CLQR plotting entry script.
- `MIMO3/plot_latest_b100_q1_ieee.py`: MIMO plotting entry script.
- `plot_series_styles.py`: shared plotting style definitions.
- `CLQR2/artifact_paths.py`, `MIMO3/artifact_paths.py`: path helpers imported by
  the plotting scripts.

## Trace notes

- CLQR figures use seed `0` results under `CLQR2/outputs`, with Fused-CPRO,
  PRCRL, SLDAC, SCAOPO, PPO-Lag, and CPO series.
- MIMO figures are stored under `MIMO3/outputs/compare/seed_0/b100_q5`.
- The MIMO Fused-CPRO files are named `b100_q5_seed0`, but their metadata records
  `run_tag=b100_q1` and `Q_update_time=1`.
- The MIMO PRCRL-labelled files are named `PRCRL_*_b100_q5_seed0`, but their
  metadata records `algorithm=HRL`; they match the corresponding HRL data.

The files were copied as-is to preserve the exact data state used by the final
figures.
