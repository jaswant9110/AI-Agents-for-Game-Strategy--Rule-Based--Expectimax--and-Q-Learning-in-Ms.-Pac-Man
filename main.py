# main.py
import gym
import numpy as np
import pandas as pd
from pacman_agents import RandomAgent, RuleBasedAgent, ExpectimaxAgent, QLearningAgent

AGENTS = {
    "Random": RandomAgent,
    "RuleBased": RuleBasedAgent,
    "Expectimax": ExpectimaxAgent,
    "QLearning": QLearningAgent
}

def run_episode(env, agent, train=False):
    observation, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    while not done:
        action = agent.choose_action(observation)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if train and hasattr(agent, "update"):
            agent.update(observation, action, reward, next_obs)
        observation = next_obs
        total_reward += reward
        steps += 1
    return total_reward, steps

def evaluate_agent(agent_class, episodes=30, train=False):
    env = gym.make("MsPacman-v0", render_mode=None) 
    agent = agent_class(env.action_space)
    rewards, steps_list = [], []
    for _ in range(episodes):
        total_reward, steps = run_episode(env, agent, train)
        rewards.append(total_reward)
        steps_list.append(steps)
    env.close()
    return rewards, steps_list

def main():
    results = {}
    for name, agent_cls in AGENTS.items():
        print(f"Evaluating {name} agent...")
        train_flag = True if name == "QLearning" else False
        rewards, steps = evaluate_agent(agent_cls, train=train_flag)
        results[name] = {"AvgReward": np.mean(rewards), "StdReward": np.std(rewards),
                         "AvgSteps": np.mean(steps), "StdSteps": np.std(steps)}
        pd.DataFrame({"Score": rewards, "Steps": steps}).to_csv(f"results_{name}.csv", index=False)
    print("\nSummary of Results:")
    for agent, metrics in results.items():
        print(f"{agent}: {metrics}")

if __name__ == "__main__":
    main()
