import json
import numpy as np


class WorkloadGenerator:

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def generate(self):
        pattern = self.config["pattern"]
        duration = self.config["duration"]
        base_rate = self.config["base_rate"]

        traffic = np.full(duration, base_rate)

        if pattern == "burst":
            start = self.config["burst_start"]
            end = self.config["burst_end"]
            burst_rate = self.config["burst_rate"]

            traffic[start:end] = burst_rate

        elif pattern == "spike":
            start = self.config["spike_start"]
            end = self.config["spike_end"]
            spike_rate = self.config["spike_rate"]

            traffic[start:end] = spike_rate

        return traffic
    

class CloudSimulator:

    def __init__(self,
                 instance_capacity=100,
                 base_latency=50):

        self.instance_capacity = instance_capacity
        self.base_latency = base_latency

    def simulate_step(self, request_rate, instances):

        capacity = instances * self.instance_capacity

        if request_rate <= capacity:
            queue = 0
            cpu_util = request_rate / capacity
            latency = self.base_latency + cpu_util * 50

        else:
            queue = request_rate - capacity
            cpu_util = 1.0
            latency = self.base_latency + 200 + queue * 0.5

        return {
            "latency": latency,
            "queue": queue,
            "cpu_util": cpu_util,
            "capacity": capacity
        }
        
if __name__ == "__main__":

    generator = WorkloadGenerator("workloads/burst.json")
    traffic = generator.generate()

    simulator = CloudSimulator()

    instances = 3

    print("\nSimulation Sample:")

    for t in range(10):
        request_rate = traffic[t]

        result = simulator.simulate_step(request_rate, instances)

        print({
            "requests": request_rate,
            "instances": instances,
            "latency": result["latency"],
            "cpu": result["cpu_util"],
            "queue": result["queue"]
        })