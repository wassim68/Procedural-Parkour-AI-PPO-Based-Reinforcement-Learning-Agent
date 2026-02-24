from stable_baselines3 import PPO
from game_state import GameState
import pygame
from collections import deque

RENDER = True  
EPISODES = 10   
FPS = 165
MODEL_PATH = "models/ppo_parkour_model_final"

env = GameState()
env.init(RENDER=RENDER) 
model = PPO.load(MODEL_PATH)

print("Testing Agent ", MODEL_PATH)
print("total agent steps : ", model.num_timesteps)

if RENDER:
    clock = pygame.time.Clock()  
    pygame.font.init()

stop_test = False

reward_history = deque(maxlen=int(0.3 * FPS)) 
total_rewards = 0
total_platforms = 0
total_survive_time = 0


for ep in range(EPISODES):
    if stop_test: break
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    ep_len = 0

    running = True
    stop_test = False
    while not done and running and not stop_test:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        ep_len += 1
        reward_history.append(reward)


        if RENDER and running:
        # Handle events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                if event.type == pygame.QUIT:
                    stop_test = True
                    running = False
            env.delta_time = clock.tick(FPS) / 1000  
            env.camera.follow(env.player.pos, env.delta_time)

            env.render(
                total_reward=total_reward,
                recent_reward=sum(reward_history) if reward_history else 0.0,
                current_action=action,
                best_platform_index=env.player.progress
            )

    print(f"Episode {ep+1}: Reward={total_reward}, Length={ep_len}")
    total_rewards += total_reward
    total_platforms += env.player.progress
    total_survive_time += ep_len / FPS
print("Testing complete.")
print(f"Average Reward over {EPISODES} episodes: {total_rewards / EPISODES}")  
print(f"Average Platforms over {EPISODES} episodes: {total_platforms / EPISODES}")  
print(f"Average Survival Time over {EPISODES} episodes: {total_survive_time / EPISODES:.2f} seconds")