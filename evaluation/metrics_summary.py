from evaluation.baselines import run_static_scaling, run_threshold_scaling
from evaluation.run_rl_policy import run_rl_policy


def compute_metrics(latencies, instances, costs):

    avg_latency = sum(latencies) / len(latencies)
    avg_instances = sum(instances) / len(instances)
    total_cost = sum(costs)

    return avg_latency, avg_instances, total_cost


def main():

    print("\nRunning experiments...\n")

    static_lat, static_inst, static_cost = run_static_scaling()
    th_lat, th_inst, th_cost = run_threshold_scaling()
    rl_lat, rl_inst, rl_cost = run_rl_policy()

    static_metrics = compute_metrics(static_lat, static_inst, static_cost)
    th_metrics = compute_metrics(th_lat, th_inst, th_cost)
    rl_metrics = compute_metrics(rl_lat, rl_inst, rl_cost)

    print("\nAUTOSCALING PERFORMANCE SUMMARY\n")

    print("{:<12} {:<15} {:<15} {:<10}".format(
        "Strategy", "Avg Latency", "Avg Instances", "Total Cost"
    ))

    print("-" * 55)

    print("{:<12} {:<15.2f} {:<15.2f} {:<10.2f}".format(
        "Static", *static_metrics
    ))

    print("{:<12} {:<15.2f} {:<15.2f} {:<10.2f}".format(
        "Threshold", *th_metrics
    ))

    print("{:<12} {:<15.2f} {:<15.2f} {:<10.2f}".format(
        "RL", *rl_metrics
    ))


if __name__ == "__main__":
    main()