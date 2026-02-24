# Procedural Parkour RL

A reinforcement learning project where an agent learns to navigate a procedurally generated 2D parkour environment using PPO (Stable-Baselines3).

The agent learns long-horizon platform traversal, gap adaptation, and timing-based actions such as jump, dash, and double jump.

---

## Features

- Custom Gymnasium environment
- Procedural platform generation (random gaps, heights, widths)
- Dense + sparse reward shaping
- PPO training 
- Real-time action visualization overlay
- Long-horizon training (600k+ steps)

---

## Environment Overview

State Space:
- Player kinematics (position, velocity)
- Action availability flags (dash, double jump)
- Relative position of upcoming platforms
- Platform geometry

Action Space:
- Idle
- Move Left
- Move Right
- Jump
- Dash

---

## Reward Design

Reward consists of:
- Forward progress shaping
- New max-distance bonus
- New platform reward
- Anti-stall penalty
- Fall penalty

---

## Training

Algorithm:
- PPO (Stable-Baselines3)

Network:
- MLP [256, 256]

Key Hyperparameters:
- learning_rate: adaptive schedule
- clip_range: 0.15
- gamma: 0.99
- gae_lambda: 0.95
- entropy coefficient: dynamic
- 
Training Strategy:
- Chunk-based training
- Evaluation after each chunk
- Best model checkpointing
- Collapse detection
  

---

## Results

Best performance achieved at:
- ~600k steps
- Stable long-horizon traversal
- High consistency
- Minimal falling


See `docs/benchmarks` for full metrics.

---

## Installation

```bash
git clone <repo>
cd procedural-parkour-rl
pip install -r requirements.txt
python test.py
