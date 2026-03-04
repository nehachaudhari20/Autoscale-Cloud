from rl.env.autoscale_env import AutoScaleEnv


def run_threshold_policy():

    env = AutoScaleEnv()

    state, _ = env.reset()

    total_reward = 0

    done = False

    while not done:

        cpu_util = state[3]

        if cpu_util > 0.7:
            action = 2  # scale up

        elif cpu_util < 0.3:
            action = 0  # scale down

        else:
            action = 1  # no change

        state, reward, done, _, _ = env.step(action)

        total_reward += reward

    print("Threshold Policy Results")
    print("Final Instances:", env.instances)
    print("Total Reward:", total_reward)


if __name__ == "__main__":
    run_threshold_policy()