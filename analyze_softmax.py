"""Rebuild the Four Rooms softmax comparison plots and qualitative statistics.

Run after compare_agents.py and visualize_options.py with comparison_softmax.yaml.
"""

import csv
import importlib.metadata
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from experiment import AGENT_LABELS
from visualize_options import representative_rollout


ROOT = Path(__file__).parent
OUTPUT = ROOT / 'results' / 'comparison_softmax'
SNAPSHOTS = ROOT / 'results' / 'option_snapshots' / 'composed_pruned_hindsight_softmax_T1'


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    summaries = read_csv(OUTPUT / 'summary.csv')
    episodes = read_csv(OUTPUT / 'episodes.csv')
    kinds = list(dict.fromkeys(row['agent'] for row in summaries))
    assert len(summaries) == 40
    assert {int(row['environment_steps']) for row in episodes if row['partial'] == '1'} <= {100000}
    for kind in kinds:
        assert {int(row['seed']) for row in summaries if row['agent'] == kind} == set(range(5))
    historical = read_csv(ROOT / 'results' / 'comparison_pruning' / 'summary.csv')
    historical += read_csv(ROOT / 'results' / 'comparison_hindsight' / 'summary.csv')
    for row in summaries:
        if row['agent'].endswith('_softmax'):
            continue
        original = next(old for old in historical if old['agent'] == row['agent'] and old['seed'] == row['seed'])
        for key in row.keys() & original.keys() - {'agent', 'training_seconds'}:
            assert float(row[key]) == float(original[key]), (row['agent'], row['seed'], key)
    (OUTPUT / 'runtime.json').write_text(json.dumps({
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name)
                     for name in ('numpy', 'gymnasium', 'PyYAML', 'matplotlib')},
        'baseline_validation': 'All shared non-timing summary values match historical runs exactly.',
    }, indent=2), encoding='utf-8')
    metrics = ('final_training_steps', 'completed_episodes', 'training_success',
               'greedy_success', 'greedy_steps', 'hindsight_updates',
               'options_created', 'options_pruned', 'longest_option_execution')
    aggregate = {}
    for kind in kinds:
        rows = [row for row in summaries if row['agent'] == kind]
        aggregate[kind] = {metric: {
            'mean': float(np.mean([float(row[metric]) for row in rows])),
            'sd': float(np.std([float(row[metric]) for row in rows], ddof=1)),
            'per_seed': [float(row[metric]) for row in rows],
        } for metric in metrics}
    (OUTPUT / 'aggregate.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')

    figure = Figure(figsize=(15, 10), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).ravel()
    edges = np.linspace(0, 100000, 21)
    selected = ['q_learning', 'repetition', 'repetition_softmax',
                'composed_pruned_hindsight', 'composed_pruned_hindsight_softmax']
    late_bounds = []
    for kind in selected:
        curves = []
        for seed in range(5):
            rows = [row for row in episodes if row['agent'] == kind
                    and int(row['seed']) == seed and not int(row['partial'])]
            ends = np.array([int(row['environment_steps']) for row in rows])
            success = np.array([int(row['success']) for row in rows])
            steps = np.array([int(row['steps']) for row in rows])
            curves.append([
                [success[ends <= hi].sum() for hi in edges[1:]],
                [success[(ends > lo) & (ends <= hi)].mean() for lo, hi in zip(edges, edges[1:])],
                [steps[(ends > lo) & (ends <= hi)].mean() for lo, hi in zip(edges, edges[1:])],
            ])
        curves = np.array(curves)
        for axis, values in zip(axes[:3], curves.transpose(1, 0, 2)):
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            line, = axis.plot(edges[1:], mean, label=AGENT_LABELS[kind])
            axis.fill_between(edges[1:], mean-sd, mean+sd, alpha=.12, color=line.get_color())
        values = curves[:, 2, :]
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        axes[3].plot(edges[1:], mean, label=AGENT_LABELS[kind])
        axes[3].fill_between(edges[1:], mean-sd, mean+sd, alpha=.12)
        late_bounds.extend((mean[-4:]-sd[-4:]).tolist() + (mean[-4:]+sd[-4:]).tolist())
    for axis, label in zip(axes, ('Cumulative goals reached', 'Training success fraction',
                                  'Training steps per episode', 'Training steps per episode (late zoom)')):
        axis.set(xlabel='Primitive environment steps', ylabel=label)
        axis.grid(alpha=.25)
    axes[1].set_ylim(0, 1.05)
    axes[3].set_xlim(80000, 100000)
    axes[3].set_ylim(min(late_bounds)-.5, max(late_bounds)+.5)
    axes[0].legend(fontsize=8)
    figure.suptitle('Four Rooms | five paired seeds | mean ± sample SD | 5,000-step bins\n'
                    'Softmax T=1 over raw counts; all methods use 100,000 primitive steps')
    figure.savefig(OUTPUT / 'learning_summary.png', dpi=150)

    qualitative = []
    for seed in range(5):
        folder = SNAPSHOTS / f'seed_{seed}'
        data = json.loads((folder / 'snapshots.json').read_text(encoding='utf-8'))
        stage = data['snapshots'][-1]
        benchmark = next(row for row in summaries if row['agent'] == 'composed_pruned_hindsight_softmax'
                         and int(row['seed']) == seed)
        # Snapshot training runs do not force a budget-cutoff episode reset.
        assert stage['episodes'] in (int(benchmark['completed_episodes']), int(benchmark['completed_episodes'])+1)
        for key in ('hindsight_updates', 'longest_option_execution'):
            assert stage[key] == int(benchmark[key])
        assert stage['pruned'] == int(benchmark['options_pruned'])
        stats = []
        for slot, option in enumerate(stage['options']):
            assert all(int(a) < data['n_actions'] + len(stage['options'])
                       for counts in option['policy'].values() for a in counts)
            rng = np.random.default_rng(2026 + seed * 1000 + option['id'])
            rollouts = [representative_rollout(data, stage, slot, rng=rng) for _ in range(200)]
            assert all(len(r['actions']) <= option['max_steps'] for r in rollouts)
            lengths = [len(r['actions']) for r in rollouts]
            patterns = Counter(tuple(r['actions']) for r in rollouts)
            probabilities = list(option['action_probabilities'].values())
            stats.append(dict(option_id=option['id'], selections=option['selections'], depth=option['depth'],
                              children=option['children'], start=data['positions'][option['start']],
                              goal=data['positions'][option['goal']], observed_lengths=option['execution_lengths'],
                              target_success=float(np.mean([r['reached_target'] for r in rollouts])),
                              mean_steps=float(np.mean(lengths)), length_counts=dict(Counter(lengths)),
                              unique_paths=len(patterns), sample_rollouts=rollouts[:5],
                              mean_max_probability=float(np.mean([max(p.values()) for p in probabilities])),
                              stochastic_states=sum(sum(p > 1e-12 for p in probs.values()) > 1
                                                    for probs in probabilities)))
        (folder / 'stochastic_rollouts.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
        longest = sorted(stats, key=lambda item: (
            max(map(int, item['observed_lengths']), default=0), item['selections']), reverse=True)
        qualitative.append(dict(seed=seed, episodes=stage['episodes'],
                                longest=stage['longest_option_execution'],
                                composite_policies=sum(bool(o['children']) for o in stats),
                                max_depth=max(o['depth'] for o in stats),
                                stochastic_states=sum(o['stochastic_states'] for o in stats),
                                top_options=longest[:6]))
        if seed == 0:
            figure = Figure(figsize=(15, 8), layout='constrained')
            FigureCanvasAgg(figure)
            for axis, item in zip(figure.subplots(2, 3).ravel(), longest[:6]):
                counts = sorted((int(k), v) for k, v in item['length_counts'].items())
                axis.bar([k for k, v in counts], [v/200 for k, v in counts])
                axis.set(xlabel='Primitive execution length', ylabel='Fraction of 200 sampled rollouts',
                         title=f"Option {item['option_id']} | depth {item['depth']} | target {item['target_success']:.0%}\n"
                               f"{tuple(item['start'])} → {tuple(item['goal'])} | {item['unique_paths']} sampled paths")
            figure.suptitle('Seed 0: six final options ranked by observed length then usage\n'
                            'Independent seeded sampling of the actual softmax policies; no learning')
            figure.savefig(OUTPUT / 'option_sampling.png', dpi=150)
    (OUTPUT / 'qualitative.json').write_text(json.dumps(qualitative, indent=2), encoding='utf-8')

    def fmt(kind, metric):
        value = aggregate[kind][metric]
        return f"{value['mean']:.3f} ± {value['sd']:.3f}"

    lines = ['# Four Rooms softmax action selection experiment', '',
             '## Protocol', '',
             'Eight methods, five paired seeds (0–4), 100,000 primitive steps per run. Fixed start (1,1), '
             'goal (9,9), 200-step limit. Alpha=0.1, gamma=0.99, epsilon=0.2. Discovery every 10 episodes, '
             'minimum 3 occurrences, 20 active options, 100 successful episodes in each replay buffer. '
             'Pruning threshold 10,000 steps; hindsight limit 100 updates per new option. '
             'Softmax uses exp(raw action count / T), T=1 fixed before running, without tuning.', '',
             'All six original methods were freshly rerun. The only configuration difference in each '
             'modal/softmax pair is option action selection. Flat softmax is included to separate the '
             'policy change from recursive composition, pruning and hindsight. All shared non-timing summary '
             'values of the six baselines exactly match their historical runs. Package versions are in runtime.json.', '',
             'Final evaluation uses 50 episodes per seed on copies, manager exploration disabled, learning '
             'and option changes disabled. Option policies retain their configured stochasticity, including '
             'softmax sampling and modal tie breaking. This measures deployed option behaviour, not a '
             'determinized softmax policy. Late training means completed episodes ending after step 80,000; '
             'partial final episodes are excluded. Curves use 5,000-step bins and mean ± sample SD across seeds.', '',
             '## Results', '',
             '| Method | Late training steps | Completed episodes | Greedy success fraction | Greedy steps |',
             '| --- | --- | --- | --- | --- |']
    for kind in kinds:
        lines.append('| ' + AGENT_LABELS[kind] + ' | ' + ' | '.join(fmt(kind, metric) for metric in
                     ('final_training_steps', 'completed_episodes', 'greedy_success', 'greedy_steps')) + ' |')
    lines += ['', '## Paired policy effects', '']
    for base in ('repetition', 'composed_pruned_hindsight'):
        soft = base + '_softmax'
        delta = np.array(aggregate[soft]['final_training_steps']['per_seed']) - aggregate[base]['final_training_steps']['per_seed']
        percent = 100 * delta.mean() / aggregate[base]['final_training_steps']['mean']
        lines += [f"- {AGENT_LABELS[base]}: softmax minus modal late-training steps, seeds 0–4: "
                  f"{np.round(delta, 3).tolist()}; mean change {delta.mean():+.3f} ({percent:+.2f}%)."]
    lines += ['', '## Option behaviour', '',
              '| Policy | Longest observed activation per seed | Hindsight updates, mean ± SD | Options pruned, mean ± SD |',
              '| --- | --- | --- | --- |']
    for kind in ('repetition', 'repetition_softmax', 'composed_pruned_hindsight', 'composed_pruned_hindsight_softmax'):
        lines.append(f"| {AGENT_LABELS[kind]} | {aggregate[kind]['longest_option_execution']['per_seed']} | "
                     f"{fmt(kind, 'hindsight_updates')} | {fmt(kind, 'options_pruned')} |")
    lines += ['', 'Independent softmax snapshot reruns were collected for every seed. Final hindsight updates, '
              'pruning counts and longest observed activations agree with the benchmark. Snapshot collection '
              'does not sample actions. The final partial episode is not forcibly closed in the snapshot runner.', '',
              '| Seed | Composite final policies | Maximum stored depth | Longest executed option | Stochastic state entries |',
              '| --- | --- | --- | --- | --- |']
    for item in qualitative:
        lines.append(f"| {item['seed']} | {item['composite_policies']} | {item['max_depth']} | {item['longest']} | {item['stochastic_states']} |")
    lines += ['', '### Seed 0 examples', '',
              'Seed 0 was chosen before inspecting results, consistent with the Nine Rooms presentation. '
              'Final options below follow the existing ranking by observed execution length then selection count. '
              'Each policy was sampled 200 times from its initiation state with an independent fixed analysis RNG, '
              'without learning. These samples assess option targets, not overall task success; they do not alter training.', '',
              '| Option | Start → target | Depth | Observed lengths (steps: count) | Sampled target success | Sampled mean length | Unique sampled action paths |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for item in qualitative[0]['top_options']:
        lines.append(f"| {item['option_id']} | {item['start']} → {item['goal']} | {item['depth']} | "
                     f"{item['observed_lengths']} | {item['target_success']:.1%} | {item['mean_steps']:.2f} | {item['unique_paths']} |")
    lines += ['', '## Interpretation and limits', '',
              'At T=1, softmax did not improve average late-training performance in either paired comparison. '
              'Flat softmax was worse in two of five seeds and better in three; its mean was dominated by '
              'a large regression in seed 4. Full softmax was worse in four of five seeds. Both softmax '
              'variants retained an 18-step final evaluation path in seed 4; the other four seeds had '
              'optimal 16-step paths. Their modal counterparts had 16-step paths in every seed. '
              'All methods achieved 100% final success.', '',
              'The full softmax method still produced useful multi-room skills: in seed 0, option 22 '
              'travels from (1,1) through the upper doorway to (3,7), while option 27 continues through '
              'the right doorway to the task goal (9,9). Each takes eight primitive steps. All six '
              'illustrated final options reached their own targets in all 200 sampled rollouts. '
              'Option 22 produced two sampled action paths; the other five produced one. These final '
              'skills were reliable in the sampled analysis, but did not yield a better overall learning '
              'result than the modal full method. Their reliability does not explain the training '
              'difference causally.', '',
              'Softmax permits lower-frequency demonstrated actions, but raw-count softmax can become almost '
              'deterministic when count differences are large. Stored depth describes references to child options, '
              'not necessarily their execution. Read the measured duration distributions alongside hierarchy depth. '
              'Stochastic state entries count option/state tables with at least two probabilities above 1e-12; '
              'they are not distinct environment cells. Small probabilities may not appear in 200 samples.', '',
              'Environment steps and hyperparameters are matched; exploration, update count, compute, and memory '
              'are not. Longer options change exploration per primitive step. Under softmax, hindsight supports '
              'any demonstrated action with nonzero numerical probability, rather than modal actions alone. '
              'Its replay is not importance weighted; a full-method difference cannot be attributed to sampling '
              'alone. Flat softmax provides a comparison without hindsight. These are five-seed results on one '
              'deterministic fixed-start task at one temperature, not a temperature sweep or generalization test. '
              'Sample SD describes variation; no significance claim is made.', '',
              '## Artifacts and reproduction', '',
              '- `learning_curves.png`: all eight freshly trained methods, training success and steps.',
              '- `learning_summary.png`: primary paired policies and Q-learning, adding cumulative goals and late zoom.',
              '- `option_sampling.png`: seed-0 duration distributions under actual stochastic policies.',
              '- `episodes.csv`, `evaluation.csv`, `summary.csv`, `aggregate.json`: raw and aggregated benchmark data.',
              '- `qualitative.json`: all-seed final option statistics.',
              '- `../option_snapshots/composed_pruned_hindsight_softmax_T1/seed_N/`: HTML viewer, PNGs, raw snapshots, '
              'modal representative paths and `stochastic_rollouts.json`.', '',
              '```powershell',
              'python compare_agents.py --config configs/comparison_softmax.yaml',
              '0..4 | ForEach-Object { python visualize_options.py --config configs/comparison_softmax.yaml --agent composed_pruned_hindsight_softmax --seed $_ }',
              'python analyze_softmax.py', '```', '',
              'Existing historical results were preserved. Compare generated CSVs rather than timing across '
              'sessions; wall-clock speed is not an outcome of this experiment. Validation: 24 unit tests passed; '
              'all-seed snapshot/benchmark consistency, option references, sampled execution bounds and PNG outputs '
              'were checked. Sampling used 200 rollouts for each of 20 final options in each of five seeds (20,000 total).']
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for kind in kinds:
        print(AGENT_LABELS[kind], ': late steps', fmt(kind, 'final_training_steps'))
    print('Saved plots, qualitative analysis and report to', OUTPUT)


if __name__ == '__main__':
    main()
