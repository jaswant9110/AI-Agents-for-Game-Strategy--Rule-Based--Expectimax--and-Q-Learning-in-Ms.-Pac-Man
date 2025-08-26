# Pac-Man AI Project

This project implements several AI agents to play Ms. Pac-Man using the OpenAI Gym environment. The agents include:

- Random Agent
- Rule-Based Agent
- Expectimax Agent
- Q-Learning Agent

## Project Structure

- `pacman_agents.py`: Contains implementations of the AI agents.
- `main.py`: Runs experiments for each agent, logging results to CSV.
- `plot_results.py`: Generates visualizations and statistical comparisons of agent performance.

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. Run experiments (this creates result CSVs):

```bash
python main.py
```

2. Generate plots and statistical comparisons:

```bash
python plot_results.py
```

## Output

- CSV logs: `results_<Agent>.csv`
- Plots: `plot_score.png`, `plot_steps.png`, `boxplot_score.png`, `boxplot_steps.png`
- Console output includes t-tests and effect sizes for performance comparison.
