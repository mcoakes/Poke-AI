from firered_rl import FireRedBattleEnv

env = FireRedBattleEnv()
observation, info = env.reset(seed=42)
done = False
total_reward = 0.0
while not done:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    total_reward += reward
    done = terminated or truncated
print(f"episode reward: {total_reward:.1f}")
env.close()

