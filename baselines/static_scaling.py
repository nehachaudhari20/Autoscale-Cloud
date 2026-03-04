from rl.env.autoscale_env import AutoScaleEnv


def run_static_policy(instances=3):

    env = AutoScaleEnv()

    state, _ = env.reset()

    env.instances = instances

    total_reward = 0

    done = False

    while not done:

        action = 1  # no scaling

        state, reward, done, _, _ = env.step(action)

        total_reward += reward

    print("Static Policy Results")
    print("Final Instances:", env.instances)
    print("Total Reward:", total_reward)


if __name__ == "__main__":
    run_static_policy()