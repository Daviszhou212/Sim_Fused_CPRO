import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def _load_array(path):
    return np.asarray(loadmat(path)["array"], dtype=np.float64).reshape(-1)


def create_comparison_plots(output_dir, run_tag, episode):
    legacy_reward = _load_array(os.path.join(output_dir, "legacy_reward_{0}.mat".format(run_tag)))
    bayesian_reward = _load_array(os.path.join(output_dir, "bayesian_reward_{0}.mat".format(run_tag)))
    legacy_cost = _load_array(os.path.join(output_dir, "legacy_cost_{0}.mat".format(run_tag)))
    bayesian_cost = _load_array(os.path.join(output_dir, "bayesian_cost_{0}.mat".format(run_tag)))
    legacy_worst_constraint = _load_array(os.path.join(output_dir, "legacy_worst_constraint_{0}.mat".format(run_tag)))
    bayesian_worst_constraint = _load_array(os.path.join(output_dir, "bayesian_worst_constraint_{0}.mat".format(run_tag)))
    x = np.arange(max(len(legacy_reward), len(bayesian_reward), len(legacy_worst_constraint), len(bayesian_worst_constraint)))

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(x[: len(legacy_reward)], legacy_reward, label="Legacy SLDAC", linewidth=2)
    axes[0].plot(x[: len(bayesian_reward)], bayesian_reward, label="Bayesian critic SLDAC", linewidth=2)
    axes[0].set_ylabel("Objective cost")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x[: len(legacy_cost)], legacy_cost, label="Legacy SLDAC", linewidth=2)
    axes[1].plot(x[: len(bayesian_cost)], bayesian_cost, label="Bayesian critic SLDAC", linewidth=2)
    axes[1].axhline(1.2, color="black", linestyle="--", linewidth=1.5, label="Average cost limit")
    axes[1].set_xlabel("Recorded episode index")
    axes[1].set_ylabel("Average delay per user")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(x[: len(legacy_worst_constraint)], legacy_worst_constraint, label="Legacy SLDAC", linewidth=2)
    axes[2].plot(x[: len(bayesian_worst_constraint)], bayesian_worst_constraint, label="Bayesian critic SLDAC", linewidth=2)
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="Worst-user residual limit")
    axes[2].set_xlabel("Recorded episode index")
    axes[2].set_ylabel("Worst-user residual")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("{0} {1}-episode comparison".format(str(run_tag), int(episode)))
    fig.tight_layout()
    plot_name = "compare_{0}_{1}ep".format(str(run_tag), int(episode))
    png_path = os.path.join(output_dir, plot_name + ".png")
    pdf_path = os.path.join(output_dir, plot_name + ".pdf")
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def create_bayesian_plots(output_dir, run_tag, episode):
    reward = _load_array(os.path.join(output_dir, "bayesian_reward_{0}.mat".format(run_tag)))
    cost = _load_array(os.path.join(output_dir, "bayesian_cost_{0}.mat".format(run_tag)))
    worst_constraint = _load_array(os.path.join(output_dir, "bayesian_worst_constraint_{0}.mat".format(run_tag)))
    x = np.arange(max(len(reward), len(cost), len(worst_constraint)))

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(x[: len(reward)], reward, label="Bayesian SLDAC tuned", linewidth=2)
    axes[0].set_ylabel("Objective cost")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x[: len(cost)], cost, label="Average cost", linewidth=2)
    axes[1].axhline(1.2, color="black", linestyle="--", linewidth=1.5, label="Average cost limit")
    axes[1].set_ylabel("Average delay per user")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(x[: len(worst_constraint)], worst_constraint, label="Worst-user residual", linewidth=2)
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="Residual limit")
    axes[2].set_xlabel("Recorded episode index")
    axes[2].set_ylabel("Worst-user residual")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("Bayesian SLDAC {0} {1}ep".format(str(run_tag), int(episode)))
    fig.tight_layout()
    plot_name = "bayesian_{0}_{1}ep".format(str(run_tag), int(episode))
    png_path = os.path.join(output_dir, plot_name + ".png")
    pdf_path = os.path.join(output_dir, plot_name + ".pdf")
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path
