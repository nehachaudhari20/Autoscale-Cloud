import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

from rl.env.simulator import WorkloadGenerator, CloudSimulator


class AutoScaleEnv(gym.Env):

    def __init__(self, workload_file="workloads/burst.json"):
        """
        Initialize the AutoScale environment.
        
        Args:
            workload_file: Path to workload JSON file. Can also be an EnvContext 
                          (from Ray RLlib), in which case we use the default.
        """
        super(AutoScaleEnv, self).__init__()

        # Handle Ray RLlib EnvContext - it passes an EnvContext object instead of string
        if not isinstance(workload_file, str):
            workload_file = "workloads/burst.json"

        # Convert to absolute path if needed
        if not os.path.isabs(workload_file):
            workload_file = os.path.join(os.getcwd(), workload_file)

        self.generator = WorkloadGenerator(workload_file)
        self.simulator = CloudSimulator()

        self.traffic = self.generator.generate()
        self.duration = len(self.traffic)

        self.current_step = 0
        self.instances = 3

        # Actions: 0=scale down, 1=no change, 2=scale up
        self.action_space = spaces.Discrete(3)

        # Observation: [request_rate, instances, latency, cpu_utilization, queue_length]
        self.observation_space = spaces.Box(
            low=0,
            high=10000,
            shape=(5,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = 0
        self.instances = 3

        request_rate = self.traffic[self.current_step]

        result = self.simulator.simulate_step(request_rate, self.instances)

        state = np.array([
            request_rate,
            self.instances,
            result["latency"],
            result["cpu_util"],
            result["queue"]
        ], dtype=np.float32)

        return state, {}

    def step(self, action):

        if action == 0:
            self.instances = max(1, self.instances - 1)

        elif action == 2:
            self.instances = min(10, self.instances + 1)

        self.current_step += 1

        # clamp current_step to valid range
        if self.current_step >= self.duration:
            self.current_step = self.duration - 1

        request_rate = self.traffic[self.current_step]

        result = self.simulator.simulate_step(request_rate, self.instances)

        state = np.array([
            request_rate,
            self.instances,
            result["latency"],
            result["cpu_util"],
            result["queue"]
        ], dtype=np.float32)

        latency_penalty = result["latency"] / 100
        cost_penalty = self.instances * 0.1

        reward = -(latency_penalty + cost_penalty)

        done = self.current_step >= self.duration - 1

        return state, reward, done, False, {}


if __name__ == "__main__":

    env = AutoScaleEnv()

    state, _ = env.reset()

    print("Initial State:", state)

    for _ in range(10):

        action = env.action_space.sample()

        state, reward, done, _, _ = env.step(action)

        print("Action:", action)
        print("State:", state)
        print("Reward:", reward)

        if done:
            break


# Register the environment with Gymnasium
gym.register(
    id="AutoScale-v0",
    entry_point="rl.env.autoscale_env:AutoScaleEnv",
)
