import argparse
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Bayesian_SLDAC_MIMO.artifact_paths import PACKAGE_ROOT, make_compare_run_id, make_run_paths
from Bayesian_SLDAC_MIMO.config import make_run_config
from Bayesian_SLDAC_MIMO.plot_compare import create_bayesian_plots, create_comparison_plots
from Bayesian_SLDAC_MIMO.sldac import run_bayesian, run_compare


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compare", "bayesian"], default="bayesian")
    parser.add_argument("--run-tag", default="b100_q1")
    parser.add_argument("--episode", type=int, default=100)
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--beta-uncertainty", type=float, default=None)
    parser.add_argument("--ensemble-init-mode", choices=["shared", "independent"], default=None)
    parser.add_argument("--critic-lr-base", type=float, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.ensemble_size is not None:
        overrides["ensemble_size"] = int(args.ensemble_size)
    if args.beta_uncertainty is not None:
        overrides["beta_uncertainty"] = float(args.beta_uncertainty)
    if args.ensemble_init_mode is not None:
        overrides["ensemble_init_mode"] = str(args.ensemble_init_mode)
    if args.critic_lr_base is not None:
        overrides["critic_lr_base"] = float(args.critic_lr_base)
    if args.smoke:
        overrides.update(
            {
                "T": 3,
                "grad_T": 3,
                "num_new_data": 2,
                "window": 10,
                "Q_update_time": 1,
                "update_time_per_episode": 1,
                "ensemble_size": int(overrides.get("ensemble_size", 2)),
            }
        )
        cfg = make_run_config(args.run_tag, episode_override=1, overrides=overrides)
        run_paths = make_run_paths("smoke_{0}".format(make_compare_run_id(args.run_tag, cfg.episode)), output_root=os.path.join(PACKAGE_ROOT, "Trash"))
    else:
        cfg = make_run_config(args.run_tag, episode_override=args.episode, overrides=overrides)
        run_paths = make_run_paths(make_compare_run_id(args.run_tag, cfg.episode), output_root=args.output_root)

    if args.mode == "compare":
        result = run_compare(cfg, output_root=run_paths.output_dir)
        png_path, pdf_path = create_comparison_plots(result.output_dir, cfg.run_tag, cfg.episode)
    else:
        result = run_bayesian(cfg, output_root=run_paths.output_dir)
        png_path, pdf_path = create_bayesian_plots(result.output_dir, cfg.run_tag, cfg.episode)
    result.summary["plots"] = {"png": png_path, "pdf": pdf_path}
    with open(os.path.join(result.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, indent=2)
    print("output_dir:", result.output_dir)
    print("plot_png:", png_path)
    print("plot_pdf:", pdf_path)


if __name__ == "__main__":
    main()
