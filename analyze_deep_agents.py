"""Summarize the two nine-room DQN comparisons without rerunning training."""
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
KINDS = ('dqn', 'outcome_dqn', 'q_learning', 'composed_pruned_hindsight')
LABELS = ('DQN', 'Outcome DQN + BC + HAR', 'Q-learning', 'Earlier composed + pruned + HAR')


def read(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    lines = ['# Nine-room DQN comparison', '',
             'Three seeds, 100,000 primitive training steps per method per seed; 100 frozen greedy evaluation episodes per seed. '
             'Sparse reward +1 at the goal, zero otherwise, 500-step time limit. Fixed task: (1,1) to (15,15). '
             'Random task: independently sampled distinct start and goal each episode. Both position and goal are observed.', '',
             'DQN: two 64-unit ReLU hidden layers, Adam 0.001, Huber loss, replay capacity 50,000, batch 64, '
             'one update per four inserted transitions, target copy every 250 updates, gamma 0.99, epsilon 0.2. '
             'Neural inputs are normalized position and goal coordinates. Tabular methods observe the equivalent joint state ID.', '',
             'Outcome discovery groups adjacent pairs of manager decisions from successful trajectories by physical endpoint displacement. '
             'The most frequent unseen nonzero displacement with at least three examples becomes an option every ten episodes, up to twelve. '
             'Each option uses a separate two-hidden-layer neural policy trained with cross entropy on the pooled demonstrations '
             '(150 minibatches, at most 256 samples each). Outputs include primitives and older options. Options initiate everywhere; '
             'they terminate at the invocation-relative displacement, their primitive-step bound, or their decision limit at a child boundary. '
             'An ancestor termination interrupts a child. Child references only point backward.', '',
             'HAR matches deterministic neural option executions against successful primitive trajectories, retaining original task rewards '
             'and discounting by primitive duration. It adds at most 100 relabelled transitions per new option to manager replay. '
             'Ordinary manager updates also use discounted semi-Markov returns. Time limits bootstrap; reaching the goal does not.', '']
    for mode in ('fixed', 'random'):
        folder = ROOT/'results'/f'dqn_nine_{mode}'
        summaries = read(folder/'summary.csv')
        episodes = read(folder/'episodes.csv')
        evaluation = read(folder/'evaluation.csv')
        assert len(summaries) == 12 and len(evaluation) == 1200
        for seed in range(3):
            paired = [[(r['start_cell'], r['goal_cell']) for r in evaluation
                       if r['agent'] == kind and int(r['seed']) == seed] for kind in KINDS]
            assert all(tasks == paired[0] for tasks in paired)
            train = [[(r['start_cell'], r['goal_cell']) for r in episodes
                      if r['agent'] == kind and int(r['seed']) == seed] for kind in KINDS]
            common = min(map(len, train))
            assert all(tasks[:common] == train[0][:common] for tasks in train)
            for kind in KINDS:
                rows = [r for r in episodes if r['agent'] == kind and int(r['seed']) == seed]
                assert int(rows[-1]['environment_steps']) == 100000
        lines += [f'## {mode.capitalize()} task', '',
                  'Mean ± sample standard deviation across seeds. Evaluation steps include failures at the time limit.', '',
                  '| Method | Greedy success | Greedy steps | Final training success | Options | HAR transitions | Max depth |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for kind, label in zip(KINDS, LABELS):
            rows = [r for r in summaries if r['agent'] == kind]
            def fmt(key, percent=False):
                vals = np.array([float(r[key]) for r in rows])*(100 if percent else 1)
                return f'{vals.mean():.1f} ± {vals.std(ddof=1):.1f}'+('%' if percent else '')
            lines.append('| '+ ' | '.join([label, fmt('greedy_success', True), fmt('greedy_steps'),
                         fmt('final_training_success', True), fmt('options'), fmt('hindsight_updates'), fmt('max_option_depth')])+' |')
        lines += ['', f'![Learning curves](dqn_nine_{mode}/learning_curves.png)', '']
    lines += ['## Interpretation limits', '',
              'These are modest-budget tests on one layout, without hyperparameter tuning. Equal environment steps do not imply equal compute: '
              'HAR inserts extra transitions and triggers extra gradient updates. The earlier method uses tabular cloning with state-specific '
              'initiation and pruning, so this is a method comparison rather than an isolated neural-cloning ablation. '
              'Fixed-task evaluation repeats one deterministic task; random-task evaluation covers 300 matched sampled tasks across the seeds. '
              'The training task streams match by episode index, while different episode lengths produce different numbers of observed tasks. '
              'A low success rate at this budget is not evidence of convergence. Timing includes overlapping jobs and should not be used to rank methods.']
    (ROOT/'results'/'dqn_nine_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
