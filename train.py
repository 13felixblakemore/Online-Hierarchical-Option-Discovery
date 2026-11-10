"""Train a tabular agent in Four Rooms or Nine Rooms.

Usage:
    pip install -r requirements.txt
    python train.py --config configs/config.yaml
    python train.py --config configs/config_nine_rooms.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from agents import QLearningAgent
from experiment import make_agent, make_env, update_agent, write_csv


def train(env, agent, episodes, seed, log_every):
    """Run training episodes and return per-episode statistics."""
    history = []

    for episode in range(1, episodes + 1):
        # Seed once, so random starts still vary between episodes.
        state, _ = env.reset(seed=seed if episode == 1 else None)
        total_reward = 0.0
        steps = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            update_agent(agent, state, action, reward, next_state, terminated, truncated)

            total_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break

        history.append({
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "success": int(terminated),
        })

        if episode % log_every == 0 or episode == episodes:
            recent = history[-log_every:]
            mean_reward = np.mean([row["reward"] for row in recent])
            success_rate = np.mean([row["success"] for row in recent])
            print(f"Episode {episode:5d}/{episodes} | "
                  f"mean reward: {mean_reward:.2f} | "
                  f"success: {success_rate:.0%}")

    return history


def save_learning_curves(history, path, window=100):
    """Save trailing episode averages as a PNG, without opening a window."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("learning_curve_window must be a positive integer")
    if not history:
        raise ValueError("Cannot plot an empty training history")

    episodes = np.array([row["episode"] for row in history])
    ends = np.arange(1, len(history) + 1)
    starts = np.maximum(0, ends - window)
    figure = Figure(figsize=(12, 3.5), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, (key, label) in zip(figure.subplots(1, 3), (
        ("reward", "Episode reward"), ("success", "Success rate"),
        ("steps", "Steps per episode"),
    )):
        values = np.array([row[key] for row in history], dtype=float)
        totals = np.concatenate(([0.0], np.cumsum(values)))
        axis.plot(episodes, (totals[ends] - totals[starts]) / (ends - starts))
        axis.set(xlabel="Episode", ylabel=label)
        axis.grid(alpha=0.3)
        if key == "success":
            axis.set_ylim(0, 1)
    figure.suptitle(f"Training learning curves (trailing mean, window={window})")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description="Train an agent in a room environment")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "configs" / "config.yaml")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config.get("seed", 42)
    settings = config.get("training", {})
    episodes = settings.get("episodes", 1000)
    log_every = settings.get("log_every", 100)
    if episodes < 1 or log_every < 1:
        raise ValueError("episodes and log_every must be positive")

    env = make_env(config.get("environment", {}))
    agent = make_agent(config["agent"], env, seed)

    try:
        history = train(env, agent, episodes, seed, log_every)
    finally:
        env.close()

    # Save results relative to the config file's folder.
    output = config.get("output", {})
    if output.get("csv"):
        path = args.config.parent / output["csv"]
        write_csv(path, history)
        print(f"Saved episode results to {path}")

    if isinstance(agent, QLearningAgent) and output.get("q_table"):
        path = args.config.parent / output["q_table"]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, agent.q)
        print(f"Saved Q-table to {path}")

    if output.get("learning_curves", False):
        path = (Path(__file__).parent / "results"
                / f"{args.config.stem}_learning_curves.png")
        save_learning_curves(history, path, output.get("learning_curve_window", 100))
        print(f"Saved learning curves to {path}")


if __name__ == "__main__":
    main()
