import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from game_state import GameState  
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

TOTAL_TIMESTEPS = 1_000_000
CHUNK_SIZE = 100_000
EVAL_EPISODES = 20

MODEL_PATH = "models_PPO/ppo_parkour_model"


# environment
env = GameState()
check_env(env)

# check if a model already exists
if os.path.isfile(MODEL_PATH + ".zip"):
    print("Loading existing model...")
    model = PPO.load(MODEL_PATH + ".zip", env=env)
else:
    print("No existing model found. Creating a new one")
    policy_kwargs = dict(
    net_arch=[256, 256]
    )

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

#create backup 
model.save(MODEL_PATH + "_backup.zip")

class MonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_entropy = None

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        logs = self.model.logger.name_to_value

        approx_kl = logs.get("train/approx_kl", None)
        entropy = logs.get("train/entropy_loss", None)
        value_loss = logs.get("train/value_loss", None)
        explained_var = logs.get("train/explained_variance", None)

        print("\n--- Training Diagnostics ---")
        print(f"KL Divergence:        {approx_kl}")
        print(f"Entropy Loss:         {entropy}")
        print(f"Value Loss:           {value_loss}")
        print(f"Explained Variance:   {explained_var}")
        print("----------------------------\n")

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

    model.learn(
        total_timesteps=CHUNK_SIZE,
        reset_num_timesteps=False,
        callback=MonitorCallback()
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
        model.save(MODEL_PATH+".zip")
        print("New best model saved.")
    else:
        print("No improvement.")

    model.save(f'{MODEL_PATH}_step_{timesteps_done}.zip')

    # collapse detection
    if mean_reward < -20:
        print("Reward collapsed, Stopping early.")
        break

print("Training complete.")