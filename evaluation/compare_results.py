import matplotlib.pyplot as plt

from evaluation.baselines import run_static_scaling, run_threshold_scaling
from evaluation.run_rl_policy import run_rl_policy


static_lat, static_inst, static_cost = run_static_scaling()
th_lat, th_inst, th_cost = run_threshold_scaling()
rl_lat, rl_inst, rl_cost = run_rl_policy()


# Latency comparison
plt.figure()
plt.plot(static_lat, label="Static")
plt.plot(th_lat, label="Threshold")
plt.plot(rl_lat, label="RL")
plt.title("Latency Comparison")
plt.legend()
plt.show()


# Instance comparison
plt.figure()
plt.plot(static_inst, label="Static")
plt.plot(th_inst, label="Threshold")
plt.plot(rl_inst, label="RL")
plt.title("Instance Count Comparison")
plt.legend()
plt.show()


# Cost comparison
plt.figure()
plt.plot(static_cost, label="Static")
plt.plot(th_cost, label="Threshold")
plt.plot(rl_cost, label="RL")
plt.title("Cost Comparison")
plt.legend()
plt.show()