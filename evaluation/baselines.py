import numpy as np
from rl.env.autoscale_env import AutoScaleEnv


def run_static_scaling():

    env = AutoScaleEnv()

    state, _ = env.reset()

    done = False

    latencies = []
    instances = []
    costs = []

    fixed_instances = 3

    while not done:

        env.instances = fixed_instances

        state, reward, done, _, _ = env.step(1)

        latencies.append(state[2])
        instances.append(env.instances)

        cost = env.instances * 0.1
        costs.append(cost)

    return latencies, instances, costs


def run_threshold_scaling():

    env = AutoScaleEnv()

    state, _ = env.reset()

    done = False

    latencies = []
    instances = []
    costs = []

    while not done:

        cpu = state[3]

        if cpu > 0.7:
            action = 2
        elif cpu < 0.3:
            action = 0
        else:
            action = 1

        state, reward, done, _, _ = env.step(action)

        latencies.append(state[2])
        instances.append(env.instances)

        cost = env.instances * 0.1
        costs.append(cost)

    return latencies, instances, costs