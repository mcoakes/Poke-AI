import argparse

from firered_rl import make_vector_env

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=4, choices=range(1, 5))
args = parser.parse_args()
envs = make_vector_env(args.num_envs)
observations, infos = envs.reset(seed=42)
for _ in range(100):
    observations, rewards, terminations, truncations, infos = envs.step(
        envs.action_space.sample()
    )
print(f"ran {args.num_envs} environments")
envs.close()

