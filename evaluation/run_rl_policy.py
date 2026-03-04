import os
import ray

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO

from rl.env.autoscale_env import AutoScaleEnv


def env_creator(config):
    return AutoScaleEnv()


def run_rl_policy():
    import torch

    ray.init(ignore_reinit_error=True)

    register_env("autoscale_env", env_creator)

    checkpoint_path = os.path.abspath("models/ppo_autoscale")

    algo = PPO.from_checkpoint(checkpoint_path)

    module = algo.get_module()

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