import os
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from game_state import GameState

TOTAL_TIMESTEPS = 1_000_000
CHUNK_SIZE = 30_000
EVAL_EPISODES = 20

MODEL_PATH = "models/ppo_parkour_model"
LOG_FILE = "docs/benchmarks/training_metrics.csv"

env = GameState()
check_env(env)

if os.path.isfile(MODEL_PATH + ".zip"):
    print("Loading existing model...")
    model = PPO.load(MODEL_PATH + ".zip", env=env)
else:
    print("No existing model found. Creating a new one")
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.02,
        verbose=2
    )

model.save(MODEL_PATH + "_backup.zip")

if not os.path.isfile(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timesteps",
            "mean_reward",
            "std_reward",
            "best_mean_reward",
            "learning_rate",
            "entropy_coef",
            "approx_kl",
            "value_loss",
            "explained_variance"
        ])

class MonitorCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.last_logs = {}

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        logs = self.model.logger.name_to_value
        self.last_logs = {
            "approx_kl": logs.get("train/approx_kl"),
            "value_loss": logs.get("train/value_loss"),
            "explained_variance": logs.get("train/explained_variance")
        }
        return True


timesteps_done = model.num_timesteps

best_mean_reward, best_std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=True
)

while timesteps_done < TOTAL_TIMESTEPS:

    progress_ratio = timesteps_done / TOTAL_TIMESTEPS

    current_lr = 3e-4 * (1 - progress_ratio)
    model.learning_rate = current_lr

    model.ent_coef = 0.01 + 0.02 * progress_ratio

    print(f"\nTraining chunk starting at step {timesteps_done}")
    print(f"Current LR: {current_lr}")
    print(f"Entropy Coef: {model.ent_coef}")

    callback = MonitorCallback()

    model.learn(
        total_timesteps=CHUNK_SIZE,
        reset_num_timesteps=False,
        callback=callback
    )

    timesteps_done += CHUNK_SIZE

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True
    )

    print(f"\nEvaluation after {timesteps_done} steps:")
    print(f"Mean Reward: {mean_reward}")
    print(f"Std Reward:  {std_reward}")

    if mean_reward > best_mean_reward:
        best_mean_reward = mean_reward
        model.save(MODEL_PATH + ".zip")
        print("New best model saved.")
    else:
        print("No improvement.")

    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timesteps_done,
            mean_reward,
            std_reward,
            best_mean_reward,
            current_lr,
            model.ent_coef,
            callback.last_logs.get("approx_kl"),
            callback.last_logs.get("value_loss"),
            callback.last_logs.get("explained_variance")
        ])

    model.save(f"{MODEL_PATH}_step_{timesteps_done}.zip")

    if mean_reward < -20:
        print("Reward collapsed, stopping early.")
        break

print("Training complete.")