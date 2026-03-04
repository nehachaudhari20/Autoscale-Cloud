import os
import ray
import torch

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from rl.env.autoscale_env import AutoScaleEnv


def env_creator(config):
    return AutoScaleEnv()


def run_rl_policy():

    # Start Ray (keep memory small for t2.micro)
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=100 * 1024 * 1024,
        _memory=300 * 1024 * 1024
    )

    # Register environment
    register_env("autoscale_env", env_creator)

    # Absolute checkpoint path
    checkpoint_path = os.path.abspath("models/ppo_autoscale")

    # Build PPO algorithm
    config = (
        PPOConfig()
        .environment("autoscale_env")
        .framework("torch")
    )

    algo = config.build()

    # Restore trained weights
    algo.restore(checkpoint_path)

    # Get RL module
    module = algo.get_module()

    # Create environment
    env = AutoScaleEnv()

    state, _ = env.reset()

    done = False

    latencies = []
    instances = []
    costs = []

    while not done:

        obs_tensor = torch.tensor([state], dtype=torch.float32)

        output = module.forward_inference({"obs": obs_tensor})

        logits = output["action_dist_inputs"]

        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample().item()

        state, reward, done, _, _ = env.step(action)

        latencies.append(state[2])
        instances.append(env.instances)

        cost = env.instances * 0.1
        costs.append(cost)

    return latencies, instances, costs