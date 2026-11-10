"""Analyze the fixed-budget random-start/random-goal Four Rooms comparison."""

import csv
import importlib.metadata
import json
import platform
from collections import deque
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from experiment import AGENT_LABELS
from four_rooms import FourRoomsEnv


ROOT = Path(__file__).parent
OUTPUT = ROOT / 'results' / 'comparison_random_goals'
KINDS = ('q_learning', 'composed_pruned_hindsight')


def read_csv(name):
    with (OUTPUT / name).open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def shortest_distances(env):
    distances = {}
    for source in env.positions:
        found = {source: 0}
        queue = deque([source])
        while queue:
            row, col = queue.popleft()
            for dr, dc in env.MOVES.values():
                target = row+dr, col+dc
                if target in env.position_to_state and target not in found:
                    found[target] = found[row, col] + 1
                    queue.append(target)
        for target, distance in found.items():
            distances[env.position_to_state[source], env.position_to_state[target]] = distance
    return distances


def main():
    episodes = read_csv('episodes.csv')
    evaluation = read_csv('evaluation.csv')
    summaries = read_csv('summary.csv')
    assert len(summaries) == 10 and len(evaluation) == 10000
    env = FourRoomsEnv()
    distances = shortest_distances(env)
    env.close()
    for row in evaluation:
        distance = distances[int(row['start_cell']), int(row['goal_cell'])]
        assert distance > 0
        row['shortest_steps'] = distance
        row['spl'] = int(row['success']) * distance / max(distance, int(row['steps']))
        row['optimal'] = int(int(row['success']) and int(row['steps']) == distance)
    with (OUTPUT / 'evaluation_paths.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=evaluation[0])
        writer.writeheader()
        writer.writerows(evaluation)

    diagnostics = []
    for seed in range(5):
        train = [[r for r in episodes if r['agent'] == kind and int(r['seed']) == seed]
                 for kind in KINDS]
        tasks = [[(r['start_cell'], r['goal_cell']) for r in rows] for rows in train]
        common = min(map(len, tasks))
        assert tasks[0][:common] == tasks[1][:common]
        test = [[(r['start_cell'], r['goal_cell']) for r in evaluation
                 if r['agent'] == kind and int(r['seed']) == seed] for kind in KINDS]
        assert test[0] == test[1]
        for kind, rows in zip(KINDS, train):
            assert int(rows[-1]['environment_steps']) == 1000000
            assert all(r['start_cell'] != r['goal_cell'] for r in rows)
            diagnostics.append(dict(agent=kind, seed=seed, tasks_seen=len(rows),
                                    unique_pairs=len({(r['start_cell'], r['goal_cell']) for r in rows}),
                                    unique_goals=len({r['goal_cell'] for r in rows})))
            assert diagnostics[-1]['unique_goals'] == 104

    aggregate = {}
    for kind in KINDS:
        per_seed = []
        for seed in range(5):
            summary = next(r for r in summaries if r['agent'] == kind and int(r['seed']) == seed)
            rows = [r for r in evaluation if r['agent'] == kind and int(r['seed']) == seed]
            successful = [r for r in rows if int(r['success'])]
            training = [r for r in episodes if r['agent'] == kind and int(r['seed']) == seed and not int(r['partial'])]
            per_seed.append(dict(
                seed=seed,
                evaluation_success=float(np.mean([int(r['success']) for r in rows])),
                evaluation_steps=float(np.mean([int(r['steps']) for r in rows])),
                evaluation_spl=float(np.mean([r['spl'] for r in rows])),
                evaluation_optimal=float(np.mean([r['optimal'] for r in rows])),
                successful_path_ratio=float(np.mean([int(r['steps'])/r['shortest_steps'] for r in successful]))
                                      if successful else float('nan'),
                training_goals=sum(int(r['success']) for r in training),
                **{key: float(summary[key]) for key in (
                    'completed_episodes', 'final_training_success', 'final_training_steps',
                    'options_created', 'options_pruned', 'hindsight_updates',
                    'longest_option_execution', 'max_option_depth')},
            ))
        aggregate[kind] = {'per_seed': per_seed, 'metrics': {
            key: dict(mean=float(np.mean([r[key] for r in per_seed])),
                      sd=float(np.std([r[key] for r in per_seed], ddof=1)))
            for key in per_seed[0] if key != 'seed'}}
    (OUTPUT / 'aggregate.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    (OUTPUT / 'task_validation.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    (OUTPUT / 'runtime.json').write_text(json.dumps(dict(
        python=platform.python_version(), packages={name: importlib.metadata.version(name)
        for name in ('gymnasium', 'numpy', 'PyYAML', 'matplotlib')},
        training_tasks='Same task sequence prefix within each paired seed.',
        evaluation_tasks='All 1,000 evaluation tasks match within each paired seed.',
    ), indent=2), encoding='utf-8')

    figure = Figure(figsize=(14, 9), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).ravel()
    edges = np.linspace(0, 1000000, 21)
    for kind in KINDS:
        curves = []
        for seed in range(5):
            rows = [r for r in episodes if r['agent'] == kind and int(r['seed']) == seed and not int(r['partial'])]
            ends = np.array([int(r['environment_steps']) for r in rows])
            successes = np.array([int(r['success']) for r in rows])
            steps = np.array([int(r['steps']) for r in rows])
            optimal = np.array([distances[int(r['start_cell']), int(r['goal_cell'])] for r in rows])
            spl = successes * optimal / np.maximum(optimal, steps)
            curves.append([
                [successes[(ends > lo) & (ends <= hi)].mean() for lo, hi in zip(edges, edges[1:])],
                [steps[(ends > lo) & (ends <= hi)].mean() for lo, hi in zip(edges, edges[1:])],
                [successes[ends <= hi].sum() for hi in edges[1:]],
                [spl[(ends > lo) & (ends <= hi)].mean() for lo, hi in zip(edges, edges[1:])],
            ])
        for axis, values in zip(axes, np.array(curves).transpose(1, 0, 2)):
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            line, = axis.plot(edges[1:], mean, label=AGENT_LABELS[kind])
            axis.fill_between(edges[1:], mean-sd, mean+sd, color=line.get_color(), alpha=.15)
    for axis, label in zip(axes, ('Training success fraction', 'Training steps per episode',
                                  'Cumulative goals reached', 'Success weighted by path efficiency (SPL)')):
        axis.set(xlabel='Primitive environment steps', ylabel=label)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[3].set_ylim(0, 1.05)
    figure.suptitle('Four Rooms: random start and goal each episode | both positions observed\n'
                    'Five paired seeds | 1,000,000 steps each | mean ± sample SD | 50,000-step bins')
    figure.savefig(OUTPUT / 'learning_summary.png', dpi=150)

    figure = Figure(figsize=(12, 4), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for kind in KINDS:
        seed_curves = []
        for seed in range(5):
            rows = [r for r in evaluation if r['agent'] == kind and int(r['seed']) == seed]
            seed_curves.append([[np.mean([r[key] if key == 'spl' else int(r[key]) for r in rows
                                         if lo <= r['shortest_steps'] < hi])
                                 for lo, hi in ((1,5), (5,9), (9,13), (13,17), (17,30))]
                                for key in ('success', 'spl')])
        for axis, values in zip(axes, np.array(seed_curves).transpose(1, 0, 2)):
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            axis.errorbar(range(5), mean, yerr=sd, marker='o', capsize=3, label=AGENT_LABELS[kind])
    for axis, label in zip(axes, ('Evaluation success fraction', 'Evaluation SPL')):
        axis.set(xticks=range(5), xticklabels=['1–4','5–8','9–12','13–16','17+'],
                 xlabel='Shortest path length of sampled task', ylabel=label, ylim=(0,1.05))
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    figure.suptitle('Frozen final policies | 1,000 matched random tasks per seed | mean ± sample SD')
    figure.savefig(OUTPUT / 'evaluation_by_distance.png', dpi=150)

    def fmt(kind, key, percent=False):
        item = aggregate[kind]['metrics'][key]
        factor = 100 if percent else 1
        suffix = '%' if percent else ''
        return f"{factor*item['mean']:.3f} ± {factor*item['sd']:.3f}{suffix}"

    lines = ['# Random-start, random-goal Four Rooms comparison', '', '## Setup', '',
             'Q-learning versus composed + pruned + hindsight, most-frequent option policy. '
             'Five paired seeds (0–4), 1,000,000 primitive environment steps per method/seed, fixed before running. '
             'The budget is larger than the earlier fixed-goal experiments because the joint state space '
             'has 10,816 states. No tuning or budget extension after viewing results.', '',
             'Both endpoints are sampled uniformly among distinct walkable cells on every reset: goal uniformly '
             'among 104 cells, start uniformly among the remaining 103. Goals stay fixed within an episode. '
             'Observations encode position_cell * 104 + goal_cell; divmod(observation,104) recovers both cell IDs, '
             'which map to (row,column) through the environment positions list. Both agents receive the same '
             'joint observation and tabular representation. No shortest-path information is given to either agent.', '',
             'Reward is 1 on reaching the sampled goal and otherwise 0. Episode limit 200; truncations bootstrap. '
             'Alpha=0.1, gamma=0.99, epsilon=0.2. Option settings match the earlier full method: 20 active slots '
             'globally, discovery every 10 episodes, minimum 3 occurrences, 100-successful-episode buffers, '
             '10,000-step inactivity pruning, at most 100 hindsight updates per new option.', '',
             'Both methods see the same seeded training task sequence by episode number; different episode '
             'lengths mean different numbers of tasks within the step budget. This prefix agreement was checked '
             'for all seeds. Final evaluation uses 1,000 exactly matched random tasks per seed with manager '
             'exploration disabled and no learning, discovery, pruning or hindsight on copied agents. Modal '
             'option ties and Q-value ties retain random tie breaking. Evaluation seed formula is '
             '10000 + 100*training_seed + episode_index; panels overlap across training seeds and repeated '
             'tasks are possible. Seed SD is across trained agents, not 5,000 independent evaluation panels.', '',
             '## Results', '', '| Metric (mean ± sample SD across five seeds) | Q-learning | Composed + pruned + hindsight |',
             '| --- | --- | --- |']
    for key, label, percent in (
        ('evaluation_success', 'Final evaluation success', True),
        ('evaluation_steps', 'Evaluation steps including failures (200-step cap)', False),
        ('evaluation_spl', 'Evaluation SPL', False),
        ('evaluation_optimal', 'Fraction of evaluation tasks solved optimally', True),
        ('successful_path_ratio', 'Path length / shortest path, successful evaluations only', False),
        ('final_training_success', 'Late-training success', True),
        ('final_training_steps', 'Late-training steps per episode', False),
        ('training_goals', 'Total training goals reached', False),
        ('completed_episodes', 'Completed training episodes', False),
        ('options_created', 'Options created', False),
        ('options_pruned', 'Options pruned', False),
        ('hindsight_updates', 'Extra manager hindsight updates', False),
    ):
        lines.append(f'| {label} | {fmt(KINDS[0], key, percent)} | {fmt(KINDS[1], key, percent)} |')
    delta = [b['evaluation_success'] - a['evaluation_success'] for a, b in zip(
        aggregate[KINDS[0]]['per_seed'], aggregate[KINDS[1]]['per_seed'])]
    baseline = aggregate[KINDS[0]]['metrics']
    option_metrics = aggregate[KINDS[1]]['metrics']
    lines += ['', f'Paired final success changes (option agent minus Q-learning), percentage points, seeds 0–4: '
              f'{np.round(np.array(delta)*100, 2).tolist()}.', '',
              f"The option agent changes mean final success by {100*np.mean(delta):+.3f} percentage points, "
              f"evaluation SPL by {option_metrics['evaluation_spl']['mean']-baseline['evaluation_spl']['mean']:+.4f}, "
              f"and total training goals reached by {option_metrics['training_goals']['mean']-baseline['training_goals']['mean']:+.1f} "
              'relative to Q-learning. These describe the measured runs, without a significance claim.', '',
              'Late training uses completed episodes ending in the last 20% of the budget. Partial final '
              'episodes are excluded. SPL is success * shortest_path / max(shortest_path, actual_steps); '
              'failures score zero. Shortest paths are computed by BFS for reporting only. Success-only path '
              'ratios can select different task subsets, so SPL and paired success are the primary evaluation '
              'measures. Evaluation is in-distribution across all random tasks, not a held-out-goal or '
              'held-out-layout generalization test.', '', '## Option behaviour and interpretation', '',
              'This experiment uses the existing option algorithm on the joint state space without a new '
              'cross-goal abstraction. Option initiation, target, cloned policy and hindsight matching '
              'include the episode goal. Replaying a trajectory for a different task goal cannot match an '
              'option initiation. Rewards keep their original meaning; no task goals or rewards are relabelled.', '',
              'This makes each discovered option task-goal-specific. The 20 active slots are shared across '
              '104 possible goals; they are not 20 per goal. Sparse repeated joint-state pairs, successful-only '
              'buffers and the global capacity may constrain discovery and reuse, but this comparison does '
              'not isolate those mechanisms. Plain tabular Q-learning also has no generalization between '
              'goal columns. This is a direct comparison of the existing algorithms with sufficient goal '
              'information, not a goal-independent skill-transfer method.', '',
              'Longest observed top-level option executions, per seed: '
              f"{[int(r['longest_option_execution']) for r in aggregate[KINDS[1]]['per_seed']]}. "
              'Maximum final stored hierarchy depths: '
              f"{[int(r['max_option_depth']) for r in aggregate[KINDS[1]]['per_seed']]}. "
              'Stored depth is not executed duration.', '',
              f"The agent performs {option_metrics['hindsight_updates']['mean']/max(1, option_metrics['options_created']['mean']):.2f} "
              'hindsight updates per created option on average (ratio of aggregate counts), compared with '
              'the configured maximum of 100. This measures the amount of compatible retained replay '
              'available at creation, not the potential benefit of replay. Options that do not recur for '
              '10,000 primitive steps are eligible for pruning. Task-goal-specific starts and the global '
              'successful-episode buffers make cross-goal reuse unavailable in this implementation.', '',
              'Primitive environment budgets and learning parameters match. Hindsight adds replay updates '
              'and memory; compute is not matched. Manager-level epsilon yields different primitive exploration '
              'when options run. Five training seeds and a single deterministic layout limit the conclusions. '
              'No significance claim is made.', '', '## Artifacts and reproduction', '',
              '- `learning_curves.png`: training-policy success and episode lengths.',
              '- `learning_summary.png`: training curves plus cumulative goals and path efficiency.',
              '- `evaluation_by_distance.png`: frozen evaluation performance grouped by task shortest distance.',
              '- `episodes.csv` / `evaluation.csv`: raw task IDs and outcomes.',
              '- `evaluation_paths.csv`: evaluation with BFS shortest distances, optimality and SPL.',
              '- `summary.csv` / `aggregate.json`: per-seed and aggregated metrics.',
              '- `task_validation.json`: task coverage and sampling checks; `runtime.json`: package versions.', '',
              '```powershell',
              'python compare_agents.py --config configs/comparison_random_goals.yaml',
              'python analyze_random_goals.py', '```', '',
              'Validation: 28 tests passed, including joint observations, varying/reproducible task endpoints, '
              'goal consistency within episodes, actual shortest-path goal execution/reward, task pairing '
              'and hindsight rejection across task goals. The analysis also checks all training task '
              'prefixes, all paired evaluation tasks and exact step budgets.']
    (OUTPUT / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for kind in KINDS:
        print(kind, 'success', fmt(kind, 'evaluation_success', True), 'SPL', fmt(kind, 'evaluation_spl'))
    print('Saved report and plots to', OUTPUT)


if __name__ == '__main__':
    main()
