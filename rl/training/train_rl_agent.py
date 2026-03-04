import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from rl.env.autoscale_env import AutoScaleEnv


def env_creator(config):
    return AutoScaleEnv()


register_env("autoscale_env", env_creator)


def train_agent():

    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment("autoscale_env")
        .framework("torch")
        .env_runners(num_env_runners=2)
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=4000
        )
    )

    algo = config.build_algo()

    print("\nStarting Training...\n")

    for i in range(20):

        result = algo.train()

        print(f"\n===== Iteration {i} =====")

        env_metrics = result.get("env_runners", {})

        mean_reward = env_metrics.get("episode_return_mean", "N/A")
        mean_length = env_metrics.get("episode_len_mean", "N/A")

        print("Mean Reward:", mean_reward)
        print("Episode Length:", mean_length)

    # FIXED SAVE
    os.makedirs("models", exist_ok=True)
    save_path = os.path.abspath("models/ppo_autoscale")

    checkpoint = algo.save(save_path)

    print("\nModel saved at:", checkpoint)

    ray.shutdown()


if __name__ == "__main__":
    train_agent()