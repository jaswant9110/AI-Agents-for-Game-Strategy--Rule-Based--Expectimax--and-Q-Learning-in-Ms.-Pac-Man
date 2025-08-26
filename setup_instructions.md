# Setup Instructions for Pac-Man AI Project

This guide walks you through setting up and running the Pac-Man AI agents using Python and OpenAI Gym in VS Code.

---

## 1. Prerequisites

Ensure you have Python 3.8+ and [pip](https://pip.pypa.io/en/stable/) installed.
We recommend using Anaconda or a virtual environment.

---

## 2. Install Required Packages

From your terminal or VS Code integrated terminal, run:

```bash
# (Optional) Activate your conda or virtual environment
conda activate base  # or your custom env name

# Install gym with Atari support and the ROM loader
pip install "gym[atari]" autorom

# Accept the ROM license
AutoROM --accept-license

# Install other dependencies
pip install numpy pandas matplotlib scipy
```

---

## 3. Verify Gym Installation

To make sure Gym and the Ms. Pac-Man environment work, run:

```bash
python -c "import gym; env = gym.make('MsPacman-v0'); print(env.observation_space)"
```

Expected output:
```
Box(0, 255, (210, 160, 3), uint8)
```

You may see a warning about the version being out of date — you can ignore it.

---

## 4. Set VS Code Interpreter (Important)

In VS Code:

1. Press `Cmd + Shift + P` (macOS) or `Ctrl + Shift + P` (Windows/Linux)
2. Type `Python: Select Interpreter`
3. Select the interpreter path that matches:
   ```
   /Users/<yourname>/anaconda3/bin/python
   ```

---

## 5. Run the Project

### Run experiments for all agents:
```bash
python main.py
```

This will create result files like `results_QLearning.csv`.

### Plot graphs and analyze agent performance:
```bash
python plot_results.py
```

This will generate:
- Line plots (`plot_score.png`, `plot_steps.png`)
- Boxplots (`boxplot_score.png`, `boxplot_steps.png`)
- Console output with **t-test** and **Cohen's d** comparisons.

---

## 6. Output Overview

Files generated after running:
- `results_<Agent>.csv` — agent scores and survival steps
- Plots of performance trends and comparisons
- Statistical test results printed to the console

---

## Optional: Update to Latest Gym Version

To use the newest `ALE/MsPacman-v5`:
```python
env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
```

Install:
```bash
pip install gym[ale-py]
```

But for coursework, `MsPacman-v0` is fully acceptable.

---

Happy Coding!
