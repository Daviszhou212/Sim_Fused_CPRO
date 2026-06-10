import argparse
import os
import sys

import numpy as np
from scipy.io import savemat

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MultiCell_MIMO.artifact_paths import build_output_path
from MultiCell_MIMO.config import build_default_config, merge_cli_config
from MultiCell_MIMO.sldac import run_sldac


PROTECTED_CLI_FIELDS = tuple(build_default_config().keys())


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-tag", dest="run_tag", type=str, default=None)
    return parser


def main(args=None):
    parser = build_parser()
    namespace = parser.parse_args(args=args)
    config, ignored = merge_cli_config(
        build_default_config(),
        vars(namespace),
        protected_fields=PROTECTED_CLI_FIELDS,
    )
    if ignored:
        print("ignored CLI overrides:", ", ".join(sorted(ignored)))
    result = run_sldac(config)
    output_path = build_output_path(
        config["output_root"],
        "SLDAC",
        "SLDAC_multicell_{0}_seed{1}.mat".format(config["run_tag"], int(config["seed"])),
    )
    savemat(
        output_path,
        {
            "objective": np.asarray(result["objective_history"], dtype=np.float64),
            "cost": np.asarray(result["cost_history"], dtype=np.float64),
            "critic_target_mode": str(config["critic_target_mode"]),
            "actor_parameterization": str(config["actor_parameterization"]),
            "log_std_mode": str(config["log_std_mode"]),
            "seed": int(config["seed"]),
        },
    )
    print("saved:", output_path)
    return result


if __name__ == "__main__":
    main()
