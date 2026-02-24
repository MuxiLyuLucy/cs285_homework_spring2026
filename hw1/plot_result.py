import sys
import re
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb


def get_run_data(wandb_run_dir):
    run_id = re.search(r"run-\d{8}_\d{6}-(\w+)", Path(wandb_run_dir).name).group(1)
    api = wandb.Api()
    entity = api.default_entity
    run = api.run(f"{entity}/hw1-imitation/{run_id}")
    df = run.history(samples=10_000, keys=["train/loss", "eval/mean_reward"])
    if "_step" in df.columns:
        df = df.rename(columns={"_step": "step"})
    return df


def plot(df, label, output):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    train_data = df[df["train/loss"].notna()]
    ax1.plot(train_data["step"], train_data["train/loss"], lw=2)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{label} - Training Loss")
    ax1.grid(True, alpha=0.3)

    eval_data = df[df["eval/mean_reward"].notna()]
    if not eval_data.empty:
        ax2.plot(eval_data["step"], eval_data["eval/mean_reward"], lw=2, marker="o", ms=5, label="Mean Reward")
        ax2.axhline(y=0.5, color="r", ls="--", lw=2, label="Target (0.5)")
        ax2.legend()
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title(f"{label} - Evaluation Reward")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved to {output}")


if __name__ == "__main__":
    wandb_run_dir = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "Policy"
    output = sys.argv[3] if len(sys.argv) > 3 else f"training_curves_{label.lower().replace(' ', '_')}.png"

    df = get_run_data(wandb_run_dir)
    plot(df, label, output)
