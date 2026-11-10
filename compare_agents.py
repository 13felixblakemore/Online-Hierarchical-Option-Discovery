"""Reproducible equal-environment-step comparison: python compare_agents.py."""

import argparse
import copy
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from agents import RepetitionAgent
from macro_actions import MacroActionAgent
from deep_agents import DQNAgent
from experiment import (
    AGENT_LABELS,
    make_agent,
    make_env,
    update_agent,
    variant_settings,
    write_csv,
)


def run(config, kind, seed, artifact_dir=None):
    env = make_env(config['environment'])
    agent = make_agent(variant_settings(config['agent'], kind), env, seed)
    state, _ = env.reset(seed=seed)
    task = task_details(env)
    rows = []
    reward_sum = steps = 0
    start = perf_counter()
    budget = config['experiment']['environment_steps']
    try:
        for step in range(1, budget + 1):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Stop a partial final episode as a time limit, not a success.
            cutoff = step == budget and not (terminated or truncated)
            update_agent(agent, state, action, reward, next_state, terminated, truncated or cutoff)
            steps += 1
            reward_sum += reward
            state = next_state
            if terminated or truncated or cutoff:
                rows.append(dict(agent=kind, seed=seed, episode=len(rows)+1,
                                 environment_steps=step, reward=reward_sum, steps=steps,
                                 success=int(terminated), partial=int(cutoff),
                                 options=len(getattr(agent, 'options', [])), **task))
                reward_sum = steps = 0
                state, _ = env.reset()
                task = task_details(env)
        elapsed = perf_counter() - start
    finally:
        env.close()

    if artifact_dir is not None and config['experiment'].get('save_options', False) and isinstance(agent, RepetitionAgent):
        from visualize_options import snapshot

        path = artifact_dir / 'options' / f'{kind}_seed_{seed}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(positions=env.positions, walls=sorted(env.walls), moves=env.MOVES,
                    n_actions=agent.n_actions, n_cells=len(env.positions),
                    observe_goal=env.observe_goal, max_episode_steps=env.max_steps,
                    seed=seed, agent=kind, snapshot=snapshot(agent, budget))
        path.write_text(json.dumps(data), encoding='utf-8')

    if artifact_dir is not None and isinstance(agent, MacroActionAgent):
        path = artifact_dir / 'macros' / f'{kind}_seed_{seed}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(agent.snapshot()),encoding='utf-8')

    # Evaluate a copy: zero learning rate and no discovery preserve learned values.
    evaluation = copy.deepcopy(agent)
    evaluation.alpha = 0
    if isinstance(evaluation, DQNAgent):
        evaluation.learning_enabled = False
    if isinstance(evaluation, MacroActionAgent):
        evaluation.learning_enabled = False
    if isinstance(evaluation, RepetitionAgent):
        evaluation.max_options = len(evaluation.options)
        evaluation.prune_options = False
        evaluation.manager_hindsight = False
    env = make_env(config['environment'])
    eval_rows = []
    try:
        for episode in range(config['experiment']['evaluation_episodes']):
            state, _ = env.reset(seed=10000 + seed * 100 + episode)
            task = task_details(env)
            for steps in range(1, env.max_steps + 1):
                action = evaluation.select_action(state, greedy=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                if isinstance(evaluation, (RepetitionAgent, MacroActionAgent)):
                    evaluation.update(state, action, reward, next_state, terminated, truncated)
                state = next_state
                if terminated or truncated:
                    eval_rows.append(dict(agent=kind, seed=seed, episode=episode+1,
                                          success=int(terminated), steps=steps, **task))
                    break
    finally:
        env.close()
    complete = [row for row in rows if not row['partial']]
    tail = [row for row in complete if row['environment_steps'] > budget * 0.8]
    summary = dict(agent=kind, seed=seed, training_seconds=elapsed,
                   completed_episodes=len(complete),
                   training_success=np.mean([r['success'] for r in complete]),
                   final_training_success=np.mean([r['success'] for r in tail]),
                   final_training_steps=np.mean([r['steps'] for r in tail]),
                   greedy_success=np.mean([r['success'] for r in eval_rows]),
                   greedy_steps=np.mean([r['steps'] for r in eval_rows]),
                   options=len(getattr(agent, 'options', [])),
                   hindsight_updates=getattr(agent, 'hindsight_updates', 0),
                   options_created=getattr(agent, '_next_option_id', 1)-1,
                   options_pruned=sum(len(event['option_ids']) for event in getattr(agent, 'pruning_history', [])),
                   longest_option_execution=getattr(agent, 'longest_option_execution', 0),
                   max_option_depth=max((o.depth for o in getattr(agent, 'options', [])), default=0),
                   max_option_steps=max((o.max_steps for o in getattr(agent, 'options', [])), default=0))
    return rows, eval_rows, summary


def task_details(env):
    """Record random task identities without exposing extra information to agents."""
    if getattr(env, 'observe_goal', False):
        return dict(start_cell=env.position_to_state[env.agent_position],
                    goal_cell=env.position_to_state[env.goal])
    return {}


def load_baselines(config, folder):
    """Reuse recorded runs only when task, learning settings and budgets match."""
    original = yaml.safe_load((folder / 'config.yaml').read_text())
    if original['environment'] != config['environment']:
        raise ValueError('Baseline environment does not match the experiment')
    for key in ('environment_steps', 'evaluation_episodes', 'seeds'):
        if original['experiment'][key] != config['experiment'][key]:
            raise ValueError(f'Baseline experiment setting differs: {key}')
    kinds = original['experiment']['agents']
    for kind in kinds:
        if variant_settings(original['agent'], kind) != variant_settings(config['agent'], kind):
            raise ValueError(f'Baseline agent settings differ: {kind}')
    data = {}
    for filename in ('episodes', 'evaluation', 'summary'):
        groups = {}
        with (folder / f'{filename}.csv').open(newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                converted = {}
                for key, value in row.items():
                    if key == 'agent':
                        converted[key] = value
                    else:
                        try:
                            converted[key] = int(value)
                        except ValueError:
                            converted[key] = float(value)
                groups.setdefault((row['agent'], int(row['seed'])), []).append(converted)
        data[filename] = groups
    return data


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root/'configs'/'comparison.yaml')
    args = parser.parse_args()
    config_path = args.config
    config = yaml.safe_load(config_path.read_text())
    baseline_name = config['experiment'].get('reuse_baselines_from')
    baselines = load_baselines(config, root / 'results' / baseline_name) if baseline_name else None
    output = root / 'results' / config_path.stem
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(config))
    histories, evaluations, summaries = [], [], []
    figure = Figure(figsize=(12, 4), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    budget = config['experiment']['environment_steps']
    edges = np.linspace(0, budget, 21)
    for kind in config['experiment'].get('agents', ['q_learning', 'repetition']):
        label = AGENT_LABELS[kind]
        curves = []
        for seed in config['experiment']['seeds']:
            key = kind, seed
            if baselines is not None and key in baselines['summary']:
                rows = baselines['episodes'][key]
                eval_rows = baselines['evaluation'][key]
                summary, = baselines['summary'][key]
                print(f'Reused validated baseline: {kind}, seed {seed}', flush=True)
            else:
                rows, eval_rows, summary = run(config, kind, seed, artifact_dir=output)
            histories.extend(rows)
            evaluations.extend(eval_rows)
            summaries.append(summary)
            write_csv(output / 'episodes.csv', histories)
            write_csv(output / 'evaluation.csv', evaluations)
            write_csv(output / 'summary.csv', summaries)
            curves.append([[np.mean([r[key] for r in rows if not r['partial']
                                    and lo < r['environment_steps'] <= hi])
                            for lo, hi in zip(edges, edges[1:])]
                           for key in ('success', 'steps')])
            print(json.dumps(summary), flush=True)
        curves = np.array(curves)
        for i, axis in enumerate(axes):
            mean = curves[:, i].mean(axis=0)
            sd = curves[:, i].std(axis=0, ddof=1)
            axis.plot(edges[1:], mean, label=label)
            axis.fill_between(edges[1:], mean-sd, mean+sd, alpha=0.15)
    for axis, ylabel in zip(axes, ['Training success rate', 'Steps per episode']):
        axis.set(xlabel='Primitive environment steps', ylabel=ylabel)
        axis.grid(alpha=0.3)
        axis.legend()
    axes[0].set_ylim(0, 1.05)
    figure.suptitle(f"{config['environment'].get('type', 'four_rooms')} | "
                   f"{len(config['experiment']['seeds'])} seeds | mean ± 1 SD | "
                   f"{budget//20:,}-step bins")
    figure.savefig(output / 'learning_curves.png', dpi=150)
    print(f'Results saved to {output}', flush=True)


if __name__ == '__main__':
    main()
