# reinforcement-learning

This repository contains four Mountain Car reinforcement learning scenarios, separated into each notebook. 

The scenarios compare discrete and continuous control, and they also change the objective between reaching the goal quickly and using less fuel.

## Repository structure

- [scenario_01_discrete_minsteps.ipynb](/Users/onosaeka/Uni/BCSAI/3-2/RL/reinforcement-learning/scenario_01_discrete_minsteps.ipynb)
  Discrete Mountain Car with a minimum-steps objective.
- [scenario_02_continuous_minfuel.ipynb](/Users/onosaeka/Uni/BCSAI/3-2/RL/reinforcement-learning/scenario_02_continuous_minfuel.ipynb)
  Continuous Mountain Car with a minimum-fuel objective.
- [scenario_03_discrete_minfuel.ipynb](/Users/onosaeka/Uni/BCSAI/3-2/RL/reinforcement-learning/scenario_03_discrete_minfuel.ipynb)
  Discrete Mountain Car with a minimum-fuel objective.
- [scenario_04_continuous_minsteps.ipynb](/Users/onosaeka/Uni/BCSAI/3-2/RL/reinforcement-learning/scenario_04_continuous_minsteps.ipynb)
  Continuous Mountain Car with a minimum non-null actions objective.
- [utils_shared.py](/Users/onosaeka/Uni/BCSAI/3-2/RL/reinforcement-learning/utils_shared.py)
  Shared plotting and trajectory helpers used by multiple notebooks.
- [requirement.txt](/Users/onosaeka/Uni/BCSAI/3-2/RL/reinforcement-learning/requirement.txt)
  Python dependencies for the notebooks, including Gymnasium video extras.
- `checkpoints/`
  Saved weights, plots, and intermediate outputs created by notebook runs.
- `runs/`
  TensorBoard logs.
- `videos/`
  Saved replay videos.

## Scenario summary

### Scenario 1

Scenario 1 uses MountainCar-v0 with discrete actions.

The goal is to reach the flag in as few steps as possible. In the current notebook run, reward is effectively minus one per step, so better policies finish in fewer steps.

The notebook compares:

- Q-learning
- Expected SARSA
- DQN with a Double-DQN-style target update

### Scenario 2

Scenario 2 uses MountainCarContinuous-v0.

The goal is to reach the flag while using as little force as possible. Larger force is penalised more heavily, so the learned policy should swing smoothly instead of pushing at full strength all the time.

The main method is SAC. The notebook also contains an Expected SARSA baseline on a discretised version of the problem.

### Scenario 3

Scenario 3 goes back to the discrete action version of Mountain Car, but changes the objective to fuel saving.

Here the important question is not only whether the car reaches the goal, but also how often it uses thrust instead of coasting.

The notebook compares:

- Q-learning
- Expected SARSA
- DQN
- a hand-built fuel-aware baseline

This notebook also creates replay videos, so the Gymnasium video extras are included in the dependency file.

### Scenario 4

Scenario 4 uses continuous Mountain Car again, but changes the objective from minimum fuel to minimum non-null actions.

That means the policy should learn not only how much force to use, but also when it is worth paying the cost of taking any non-zero action at all.

The notebook compares continuous-control methods, including SAC and TD3.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```


## Main dependencies

The notebooks rely on:

- Gymnasium
- NumPy
- Matplotlib
- PyTorch
- Stable-Baselines3
- TensorBoard
- scikit-learn
- tqdm
- IPython

The Gymnasium other extra is included because Scenario 3 uses replay and video generation.

## Outputs

Running the notebooks produces several kinds of outputs:

- training curves
- evaluation tables
- policy heatmaps
- phase portraits
- Q-value or action surfaces
- visitation heatmaps
- feature importance plots
- saved checkpoints
- replay videos for the scenarios that render them

Most of these files are written into `checkpoints/`, `runs/`, or `videos/`.

