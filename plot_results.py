# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

AGENTS = ["Random", "RuleBased", "Expectimax", "QLearning"]

# Moving average plots
for metric in ["Score", "Steps"]:
    plt.figure(figsize=(10, 6))
    for agent in AGENTS:
        df = pd.read_csv(f"results_{agent}.csv")
        series = df[metric].rolling(window=3).mean()
        plt.plot(series, label=agent)
    plt.title(f"{metric} per Episode (3-Episode Moving Avg)")
    plt.xlabel("Episode")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plot_{metric}.png")
    plt.show()

# Boxplots
for metric in ["Score", "Steps"]:
    plt.figure(figsize=(10, 6))
    data = [pd.read_csv(f"results_{agent}.csv")[metric] for agent in AGENTS]
    plt.boxplot(data, labels=AGENTS)
    plt.title(f"{metric} Distribution per Agent")
    plt.ylabel(metric)
    plt.grid(True)
    plt.savefig(f"boxplot_{metric.lower()}.png")
    plt.show()

# T-tests and effect sizes
print("\nStatistical comparison against QLearning:")
ql = pd.read_csv("results_QLearning.csv")
for metric in ["Score", "Steps"]:
    print(f"\nMetric: {metric}")
    ql_data = ql[metric]
    for agent in AGENTS:
        if agent == "QLearning":
            continue
        other_data = pd.read_csv(f"results_{agent}.csv")[metric]
        t_stat, p_val = stats.ttest_ind(ql_data, other_data, equal_var=False)
        mean_diff = ql_data.mean() - other_data.mean()
        cohens_d = mean_diff / ((ql_data.std()**2 + other_data.std()**2) / 2) ** 0.5
        print(f"QLearning vs {agent}: t={t_stat:.2f}, p={p_val:.4f}, Cohen's d={cohens_d:.2f}")
