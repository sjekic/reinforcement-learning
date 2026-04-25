# utils_shared.py
# Scenario: ALL (shared utility module)
# Objective: Reusable plotting and data-collection helpers for all three Mountain Car scenarios
# Action Space: N/A
# Cost Function: N/A

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


def plot_training_curves(rewards, window=50, title='Training Rewards', save_path=None):
    """Episodic rewards with rolling moving-average overlay."""
    fig, ax = plt.subplots(figsize=(12, 5))
    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha=0.3, color='steelblue', label='Episode reward')
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax.plot(np.arange(window - 1, len(rewards)), ma,
                color='navy', lw=2, label=f'{window}-ep moving avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_policy_heatmap(get_action_fn, env, title='Learned Policy',
                        action_labels=None, n_grid=50, save_path=None):
    """2D heatmap: colour = greedy action over (position x velocity) grid."""
    pos_low, vel_low = env.observation_space.low
    pos_high, vel_high = env.observation_space.high
    pos_g = np.linspace(pos_low, pos_high, n_grid)
    vel_g = np.linspace(vel_low, vel_high, n_grid)

    grid = np.zeros((n_grid, n_grid))
    for i, p in enumerate(pos_g):
        for j, v in enumerate(vel_g):
            grid[j, i] = get_action_fn(np.array([p, v]))

    if action_labels is None:
        action_labels = ['Left (0)', 'Coast (1)', 'Right (2)']
    n_actions = len(action_labels)
    colors = ['#d62728', '#2ca02c', '#1f77b4'][:n_actions]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(grid, extent=[pos_low, pos_high, vel_low, vel_high],
              origin='lower', cmap=cmap, aspect='auto', vmin=0, vmax=n_actions - 1)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title(title)
    patches = [mpatches.Patch(color=colors[i], label=action_labels[i]) for i in range(n_actions)]
    ax.legend(handles=patches, loc='upper left')
    ax.axvline(x=0.5, color='gold', linestyle='--', lw=1.5, label='Goal (0.5)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return grid


def plot_phase_portrait(trajectories, title='Phase Portrait', save_path=None):
    """Car trajectories in (position, velocity) space, coloured by episode reward."""
    fig, ax = plt.subplots(figsize=(10, 7))
    rewards = [t['total_reward'] for t in trajectories]
    vmin, vmax = min(rewards), max(rewards)
    cmap = plt.cm.viridis
    for traj in trajectories:
        pos = [s[0] for s in traj['states']]
        vel = [s[1] for s in traj['states']]
        c = cmap((traj['total_reward'] - vmin) / max(vmax - vmin, 1e-8))
        ax.plot(pos, vel, alpha=0.4, lw=0.8, color=c)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Total episode reward')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title(title)
    ax.axvline(x=0.5, color='gold', linestyle='--', lw=1.5, label='Goal (pos=0.5)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_q_surface(q_values_fn, env, title='Q-Value Surface (max Q)',
                   n_grid=40, save_path=None):
    """3D mesh of max Q(s,a) over the (position x velocity) state space."""
    pos_low, vel_low = env.observation_space.low
    pos_high, vel_high = env.observation_space.high
    pos_g = np.linspace(pos_low, pos_high, n_grid)
    vel_g = np.linspace(vel_low, vel_high, n_grid)
    POS, VEL = np.meshgrid(pos_g, vel_g)
    Q_max = np.array([[np.max(q_values_fn(np.array([p, v])))
                       for p in pos_g] for v in vel_g])
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(POS, VEL, Q_max, cmap='coolwarm', alpha=0.85)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('max Q(s,a)')
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_state_visitation(visit_counts, env, title='State Visitation Heatmap',
                          save_path=None):
    """Log-scaled heatmap of how often each (position, velocity) cell was visited."""
    pos_low, vel_low = env.observation_space.low
    pos_high, vel_high = env.observation_space.high
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(np.log1p(visit_counts).T,
                   extent=[pos_low, pos_high, vel_low, vel_high],
                   origin='lower', cmap='hot', aspect='auto')
    plt.colorbar(im, ax=ax, label='log(1 + visits)')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title(title)
    ax.axvline(x=0.5, color='cyan', linestyle='--', lw=1.5, label='Goal')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def collect_trajectories(env, get_action_fn, n_episodes=20, max_steps=999):
    """Run n_episodes greedy episodes; return list of {states, total_reward} dicts."""
    trajectories = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep)
        states = [state.copy()]
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = get_action_fn(state)
            if np.isscalar(action):
                action = np.array([action], dtype=np.float32)
            state, reward, terminated, truncated, _ = env.step(action)
            states.append(state.copy())
            total_reward += reward
            done = terminated or truncated
            steps += 1
        trajectories.append({'states': states, 'total_reward': total_reward})
    return trajectories


def build_visit_grid(trajectories, env, n_grid=40):
    """Count state-space visits into a 2D (pos x vel) grid from trajectory data."""
    pos_low, vel_low = env.observation_space.low
    pos_high, vel_high = env.observation_space.high
    grid = np.zeros((n_grid, n_grid))
    for traj in trajectories:
        for s in traj['states']:
            pi = int(np.clip(
                (s[0] - pos_low) / (pos_high - pos_low) * (n_grid - 1), 0, n_grid - 1))
            vi = int(np.clip(
                (s[1] - vel_low) / (vel_high - vel_low) * (n_grid - 1), 0, n_grid - 1))
            grid[pi, vi] += 1
    return grid
