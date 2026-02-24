import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "docs/benchmarks/training_metrics.csv"
OUTPUT_DIR = "docs/benchmarks/plots"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

data = pd.read_csv(CSV_FILE)

timesteps = data["timesteps"]

def save_plot(y, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, y)
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

save_plot(
    data["mean_reward"],
    "Mean Reward vs Timesteps",
    "Mean Reward",
    "reward_curve.png"
)

save_plot(
    data["std_reward"],
    "Reward Std Deviation vs Timesteps",
    "Std Reward",
    "reward_std.png"
)

if "approx_kl" in data.columns:
    save_plot(
        data["approx_kl"],
        "KL Divergence vs Timesteps",
        "KL Divergence",
        "kl_divergence.png"
    )

if "value_loss" in data.columns:
    save_plot(
        data["value_loss"],
        "Value Loss vs Timesteps",
        "Value Loss",
        "value_loss.png"
    )

if "explained_variance" in data.columns:
    save_plot(
        data["explained_variance"],
        "Explained Variance vs Timesteps",
        "Explained Variance",
        "explained_variance.png"
    )

if "learning_rate" in data.columns:
    save_plot(
        data["learning_rate"],
        "Learning Rate Schedule",
        "Learning Rate",
        "lr_schedule.png"
    )

if "entropy_coef" in data.columns:
    save_plot(
        data["entropy_coef"],
        "Entropy Coefficient Schedule",
        "Entropy Coefficient",
        "entropy_schedule.png"
    )

print("plots exported to 'docs/benchmarks/plots/' directory.")