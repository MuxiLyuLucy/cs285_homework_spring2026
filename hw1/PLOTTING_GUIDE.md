# Training Curves Plotting Guide

## Usage

### Plot a single run

```bash
# From a local wandb directory
uv run python src/hw1_imitation/plot_result.py wandb/run-20260209_185148-gn0r6lgj --label "MSE Policy"

# From an exp directory
uv run python src/hw1_imitation/plot_result.py exp/seed_42_20260209_185147 --label "MSE Policy"

# From a WandB API path (entity/project/run_id)
uv run python src/hw1_imitation/plot_result.py muxi_lyu-uc-berkeley-electrical-engineering-computer-sci/hw1-imitation/gn0r6lgj --label "MSE Policy"
```

### Plot MSE vs Flow Matching comparison

```bash
uv run python src/hw1_imitation/plot_result.py \
    --mse wandb/run-XXXX-mse_run \
    --flow wandb/run-YYYY-flow_run
```

### Options

| Flag | Description |
|------|-------------|
| `--label TEXT` | Label for the single-run plot (default: "Policy") |
| `--output FILE` / `-o FILE` | Custom output path for the figure |
| `--no-show` | Save the plot without displaying it |
| `--mse PATH` | Path to MSE policy run (comparison mode) |
| `--flow PATH` | Path to Flow Matching policy run (comparison mode) |

## Output

- **Single run**: `training_curves_<label>.png` (two panels: loss + reward)
- **Comparison**: `training_curves_comparison.png` (four panels: 2x loss + 2x reward)
- **Console**: Summary statistics including final loss, reward, and target achievement
