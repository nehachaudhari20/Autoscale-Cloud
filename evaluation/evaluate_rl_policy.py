import os
import ray
import torch
import numpy as np

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO

from rl.env.autoscale_env import AutoScaleEnv


def env_creator(config):
    return AutoScaleEnv()


def evaluate_rl():

    ray.init()

    register_env("autoscale_env", env_creator)

    checkpoint_path = os.path.abspath("models/ppo_autoscale")

    algo = PPO.from_checkpoint(checkpoint_path)

    # ✅ Get RLModule (new API way)
    module = algo.get_module()

    env = AutoScaleEnv()

    obs, _ = env.reset()

    done = False
    total_reward = 0

    while not done:

        # Convert observation to tensor batch
        obs_tensor = torch.tensor([obs], dtype=torch.float32)

        # Forward inference
        output = module.forward_inference({"obs": obs_tensor})

        # ✅ Correct key for PPO categorical actions
        logits = output["action_dist_inputs"]

        # Create distribution properly
        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample().item()

        obs, reward, done, _, _ = env.step(action)

        total_reward += reward

    print("\nEvaluation Results")
    print("Total Reward:", total_reward)


if __name__ == "__main__":
    evaluate_rl()