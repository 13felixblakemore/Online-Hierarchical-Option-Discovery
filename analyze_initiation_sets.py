"""Analyze wider initiation sets and probe saved policies from every grid cell."""

import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from analyze_random_goals import shortest_distances
from experiment import AGENT_LABELS
from four_rooms import FourRoomsEnv


ROOT = Path(__file__).parent
OUTPUT = ROOT / 'results' / 'comparison_initiation_sets'
KINDS = ('q_learning', 'composed_pruned_hindsight', 'composed_pruned_hindsight_support')


def read_csv(name):
    with (OUTPUT / name).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def probe(data, slot, start, rng):
    """Run the stored policy in deterministic grid dynamics, without learning.

    Availability is deliberately bypassed to test the all-states proposal.
    Undefined policies stop immediately; no fallback navigation is supplied.
    """
    options = data['snapshot']['options']
    option = options[slot]
    count = data['n_cells']
    task_goal = option['start'] % count
    lookup = {tuple(p): i for i, p in enumerate(data['positions'])}
    state = start * count + task_goal
    stack = [dict(index=slot, steps=0, decisions=0)]
    length = 0
    while stack:
        frame = stack[-1]
        current = options[frame['index']]
        probabilities = current['action_probabilities'].get(str(state))
        if not probabilities:
            return dict(success=False, steps=length, reason='undefined_policy')
        action = int(rng.choice([int(a) for a in probabilities], p=list(probabilities.values())))
        frame['decisions'] += 1
        if action >= data['n_actions']:
            stack.append(dict(index=action-data['n_actions'], steps=0, decisions=0))
            continue
        row, col = data['positions'][state // count]
        dr, dc = data['moves'][str(action)]
        position = lookup.get((row+dr, col+dc), state // count)
        state = position * count + task_goal
        length += 1
        for frame in stack:
            frame['steps'] += 1
        for i, frame in enumerate(stack):
            current = options[frame['index']]
            if state == current['goal'] or frame['steps'] >= current['max_steps']:
                del stack[i:]
                break
        while stack:
            frame = stack[-1]
            current = options[frame['index']]
            if frame['decisions'] >= current['max_decisions'] or str(state) not in current['policy']:
                stack.pop()
            else:
                break
        if position == task_goal:
            break
    assert length <= option['max_steps']
    return dict(success=state == option['goal'], steps=length,
                reason='target' if state == option['goal'] else 'other_termination')


def main():
    summaries, episodes, evaluation = map(read_csv, ('summary.csv', 'episodes.csv', 'evaluation.csv'))
    assert len(summaries) == 15 and len(evaluation) == 15000
    env = FourRoomsEnv()
    distances = shortest_distances(env)
    grid = np.zeros((env.size, env.size))
    for row, col in env.walls:
        grid[row, col] = 1
    for seed in range(5):
        train = [[(r['start_cell'], r['goal_cell']) for r in episodes
                  if r['agent'] == kind and int(r['seed']) == seed] for kind in KINDS]
        common = min(map(len, train))
        assert all(tasks[:common] == train[0][:common] for tasks in train)
        tasks = [[(r['start_cell'], r['goal_cell']) for r in evaluation
                  if r['agent'] == kind and int(r['seed']) == seed] for kind in KINDS]
        assert all(panel == tasks[0] for panel in tasks)
    aggregate = {}
    for kind in KINDS:
        per_seed = []
        for seed in range(5):
            summary = next(r for r in summaries if r['agent'] == kind and int(r['seed']) == seed)
            train = [r for r in episodes if r['agent'] == kind and int(r['seed']) == seed]
            assert int(train[-1]['environment_steps']) == 1000000
            rows = [r for r in evaluation if r['agent'] == kind and int(r['seed']) == seed]
            optimal = [distances[int(r['start_cell']), int(r['goal_cell'])] for r in rows]
            per_seed.append(dict(
                seed=seed,
                evaluation_success=float(np.mean([int(r['success']) for r in rows])),
                evaluation_steps=float(np.mean([int(r['steps']) for r in rows])),
                evaluation_spl=float(np.mean([int(r['success'])*d/max(d,int(r['steps'])) for r,d in zip(rows,optimal)])),
                evaluation_optimal=float(np.mean([int(r['success']) and int(r['steps']) == d for r,d in zip(rows,optimal)])),
                training_goals=sum(int(r['success']) for r in train if not int(r['partial'])),
                **{key: float(summary[key]) for key in ('final_training_steps', 'completed_episodes',
                    'options_created', 'options_pruned', 'hindsight_updates', 'longest_option_execution', 'max_option_depth')},
            ))
        aggregate[kind] = dict(per_seed=per_seed, metrics={key: dict(
            mean=float(np.mean([r[key] for r in per_seed])),
            sd=float(np.std([r[key] for r in per_seed], ddof=1))) for key in per_seed[0] if key != 'seed'})
    (OUTPUT / 'aggregate.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    baseline_folder = ROOT / 'results' / 'comparison_random_goals'
    (OUTPUT / 'runtime.json').write_text(json.dumps(dict(
        python=platform.python_version(), packages={name: importlib.metadata.version(name)
            for name in ('numpy', 'gymnasium', 'PyYAML', 'matplotlib')},
        reused_baselines=str(baseline_folder),
        baseline_sha256={name: hashlib.sha256((baseline_folder/name).read_bytes()).hexdigest()
                         for name in ('config.yaml', 'episodes.csv', 'evaluation.csv', 'summary.csv')},
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
            success = np.array([int(r['success']) for r in rows])
            steps = np.array([int(r['steps']) for r in rows])
            d = np.array([distances[int(r['start_cell']), int(r['goal_cell'])] for r in rows])
            spl = success * d / np.maximum(d,steps)
            curves.append([
                [success[(ends > lo)&(ends <= hi)].mean() for lo,hi in zip(edges,edges[1:])],
                [steps[(ends > lo)&(ends <= hi)].mean() for lo,hi in zip(edges,edges[1:])],
                [success[ends <= hi].sum() for hi in edges[1:]],
                [spl[(ends > lo)&(ends <= hi)].mean() for lo,hi in zip(edges,edges[1:])],
            ])
        for axis, values in zip(axes, np.array(curves).transpose(1,0,2)):
            mean, sd = values.mean(axis=0), values.std(axis=0,ddof=1)
            line, = axis.plot(edges[1:], mean, label=AGENT_LABELS[kind])
            axis.fill_between(edges[1:],mean-sd,mean+sd,color=line.get_color(),alpha=.15)
    for axis,label in zip(axes,('Training success fraction','Training steps per episode',
                               'Cumulative goals reached','Training SPL')):
        axis.set(xlabel='Primitive environment steps',ylabel=label)
        axis.legend(fontsize=7)
        axis.grid(alpha=.25)
    axes[0].set_ylim(0,1.05)
    axes[3].set_ylim(0,1.05)
    figure.suptitle('Four Rooms: random start and task goal, both observed\n'
                    'Single-state versus demonstrated-policy initiation sets | five seeds | mean ± sample SD')
    figure.savefig(OUTPUT / 'learning_summary.png',dpi=150)

    qualitative = []
    for seed in range(5):
        data = json.loads((OUTPUT / 'options' / f'{KINDS[2]}_seed_{seed}.json').read_text())
        benchmark = next(r for r in summaries if r['agent'] == KINDS[2] and int(r['seed']) == seed)
        assert data['snapshot']['hindsight_updates'] == int(benchmark['hindsight_updates'])
        options = data['snapshot']['options']
        stats = []
        for slot, option in enumerate(options):
            initiation = set(option['initiation_states'])
            assert initiation == {int(s) for s in option['policy']} - {option['goal']}
            context = option['start'] % data['n_cells']
            assert {s % data['n_cells'] for s in initiation} == {context}
            target = option['goal'] // data['n_cells']
            rng = np.random.default_rng(20261110 + seed*1000 + option['id'])
            outcomes = []
            for cell in range(data['n_cells']):
                if cell in (context,target):
                    continue  # Neither an ended task nor an already-satisfied option is an initiation trial.
                state = cell * data['n_cells'] + context
                for _ in range(32):
                    outcome = probe(data,slot,cell,rng)
                    outcomes.append(dict(cell=cell,allowed=state in initiation,
                                         additional=state != option['start'],**outcome))
            allowed = [r for r in outcomes if r['allowed']]
            additional = [r for r in allowed if r['additional']]
            stats.append(dict(id=option['id'],slot=slot,start=option['start'],target=option['goal'],
                              initiation_states=sorted(initiation),size=len(initiation),
                              selections=option['selections'],depth=option['depth'],
                              supported_success=float(np.mean([r['success'] for r in allowed])),
                              additional_success=float(np.mean([r['success'] for r in additional])) if additional else None,
                              forced_all_states_success=float(np.mean([r['success'] for r in outcomes])),
                              undefined_start_fraction=float(np.mean([r['reason']=='undefined_policy' and r['steps']==0 for r in outcomes])),
                              per_start=[dict(cell=cell,
                                  allowed=next(r['allowed'] for r in outcomes if r['cell']==cell),
                                  success=float(np.mean([r['success'] for r in outcomes if r['cell']==cell])))
                                  for cell in sorted({r['cell'] for r in outcomes})]))
        qualitative.append(dict(seed=seed,options=stats))
        if seed == 0:
            figure = Figure(figsize=(15,9),layout='constrained')
            FigureCanvasAgg(figure)
            for axis, item in zip(figure.subplots(2,3).ravel(), sorted(stats,key=lambda r:-r['selections'])[:6]):
                axis.imshow(grid,cmap='Greys',vmin=0,vmax=1)
                for state in item['initiation_states']:
                    row,col = data['positions'][state//data['n_cells']]
                    axis.scatter(col,row,c='#24a16b',marker='s',s=180,alpha=.6)
                sr,sc = data['positions'][item['start']//data['n_cells']]
                gr,gc = data['positions'][item['target']//data['n_cells']]
                tr,tc = data['positions'][item['start']%data['n_cells']]
                axis.scatter(sc,sr,facecolors='none',edgecolors='#2266cc',s=210,linewidths=2)
                axis.scatter(gc,gr,c='#e88c22',marker='D',s=90)
                axis.scatter(tc,tr,c='#8a37b5',marker='*',s=130)
                axis.set_title(f"Option {item['id']} | {item['size']} initiation states | depth {item['depth']}\n"
                               f"Target success: {item['supported_success']:.0%} supported, {item['forced_all_states_success']:.1%} all cells",fontsize=10)
                axis.set(xticks=range(11),yticks=range(11),xlabel='Column',ylabel='Row')
                axis.tick_params(labelsize=7)
            figure.suptitle('Seed 0: six most-selected final options (fixed selection rule)\n'
                            'Green squares: initiation set | blue circle: original start | orange diamond: option target | purple star: task goal')
            figure.savefig(OUTPUT/'initiation_sets.png',dpi=150)
    (OUTPUT/'transfer_probes.json').write_text(json.dumps(qualitative,indent=2),encoding='utf-8')
    def fmt(kind,key):
        value = aggregate[kind]['metrics'][key]
        return f"{value['mean']:.3f} ± {value['sd']:.3f}"
    lines = ['# Broader initiation-set experiment', '', '## What changed', '',
             'The original singleton initiation set is a valid option initiation set, but restrictive. '
             'The new policy_support mode uses every demonstrated policy state except the option target. '
             'It does not invent actions at unseen states. Both manager availability and bootstrap maxima '
             'use the same set; hindsight matches from every allowed start and updates that actual start row. '
             'Termination limits, frequency-based discovery, cloning, pruning and manager learning are unchanged.', '',
             'This isolates expanding the initiation set over existing policy coverage. It does not introduce '
             'a new policy architecture, goal-independent skills, a model-based planner or a fallback navigator. '
             'Membership means a defined initial policy action, not a guarantee of reaching the target. '
             'Cloned policies can still terminate early or exhaust their budget; the probes below measure this.', '',
             '## Protocol', '',
             'Four Rooms, random distinct start and task goal each episode; observations include both positions '
             'as position_cell*104+goal_cell. Five paired seeds 0–4, 1,000,000 primitive steps each, 200-step '
             'episode limit. Alpha=.1, gamma=.99, epsilon=.2. Most-frequent option policy. 20 global active '
             'slots, discovery every 10 episodes, minimum 3 occurrences, buffer size 100, pruning after '
             '10,000 inactive steps, up to 100 hindsight updates per new option. These settings were fixed '
             'before running, with no tuning or budget extension.', '',
             'Q-learning and single-state baselines are reused from comparison_random_goals, not freshly trained '
             'here. The runner validates exact environment, variant settings, seed list and budgets before reuse. '
             'Remove reuse_baselines_from from the config to rerun all three. Raw baseline results are copied '
             'into this experiment; historical files remain unchanged. Timing is not compared.', '',
             'Final evaluation: 1,000 matching tasks per seed, copied agents with learning, discovery, pruning '
             'and hindsight disabled. All training task sequence prefixes and all evaluation panels were checked '
             'for agreement. Evaluation panels overlap across seeds under the existing seed formula; SD is across '
             'five trained agents. Curves are training-policy metrics in 50,000-step bins, not periodic evaluations.', '',
             '## Results', '', '| Method | Evaluation success | Evaluation steps | Evaluation SPL | Late-training steps | Training goals |',
             '| --- | --- | --- | --- | --- | --- |']
    for kind in KINDS:
        lines.append('| '+AGENT_LABELS[kind]+' | '+' | '.join(fmt(kind,key) for key in (
            'evaluation_success','evaluation_steps','evaluation_spl','final_training_steps','training_goals'))+' |')
    lines += ['', 'SPL = success * shortest_path / max(shortest_path, actual_steps). BFS is used for reporting '
              'only and is not given to the agents. Late training uses completed episodes ending in the last '
              '20% of the step budget; partial final episodes are excluded.', '',
              '| Option method | Options created | Options pruned | Hindsight updates | Longest execution per seed | Final depth per seed |',
              '| --- | --- | --- | --- | --- | --- |']
    for kind in KINDS[1:]:
        lines.append('| '+AGENT_LABELS[kind]+' | '+' | '.join(fmt(kind,key) for key in (
            'options_created','options_pruned','hindsight_updates'))+' | '+
            str([int(r['longest_option_execution']) for r in aggregate[kind]['per_seed']])+' | '+
            str([int(r['max_option_depth']) for r in aggregate[kind]['per_seed']])+' |')
    delta = [a['evaluation_spl']-b['evaluation_spl'] for a,b in zip(
        aggregate[KINDS[2]]['per_seed'],aggregate[KINDS[1]]['per_seed'])]
    lines += ['', f'Paired evaluation SPL changes, broad minus singleton: {np.round(delta,5).tolist()}.', '',
              '## Initiation and all-states probes', '',
              'For every final broad-set option, probe every physical cell under its original task-goal context. '
              'Exclude the task terminal and option target cells. Each remaining start has 32 seeded policy '
              'rollouts. Availability is bypassed for the all-states diagnostic; an undefined policy stops '
              'immediately and counts as failure. There is no fallback action, shortest-path controller, '
              'learning or extra training interaction. Dynamics are used only to execute the stored policy.', '',
              'Results below average the per-option rates within each seed. These diagnostics concern reaching '
              'an option subgoal, not whole-task success. Finite samples cannot certify success probabilities.', '',
              '| Seed | Mean initiation-set size | Target success from supported starts | From additional supported starts | Forced all-cells success | Undefined at start |',
              '| --- | --- | --- | --- | --- | --- |']
    for result in qualitative:
        opts = result['options']
        additional = [r['additional_success'] for r in opts if r['additional_success'] is not None]
        additional_rate = f'{np.mean(additional):.1%}' if additional else 'not available'
        lines.append(f"| {result['seed']} | {np.mean([r['size'] for r in opts]):.2f} | "
                     f"{np.mean([r['supported_success'] for r in opts]):.1%} | {additional_rate} | "
                     f"{np.mean([r['forced_all_states_success'] for r in opts]):.1%} | "
                     f"{np.mean([r['undefined_start_fraction'] for r in opts]):.1%} |")
    lines += ['', '## What transfers, and what does not', '',
              'A stored policy can be reused from an additional state when it already defines an action '
              'there; starting mid-demonstration can execute a useful suffix. This is reuse of demonstrated '
              'coverage, not generalization to unobserved positions. Merely exposing an option everywhere '
              'does not create the missing policy entries. An option also remains specific to its original '
              'task goal because goal identity is part of every policy key. It is unavailable under a '
              'different goal even at the same physical cell.', '',
              'To support all nonterminal positions meaningfully, a separate extension would need a policy '
              'defined across the map: for example a learned option-subgoal-conditioned controller trained '
              'from diverse starts. Its initiation set should then reflect measured reachability or success '
              'under its time budget. Relative features can share behaviour across positions, but walls '
              'and doorways mean a translated action sequence need not remain valid. That architecture '
              'change is deliberately excluded from this initiation-only ablation.', '',
              'The 20-option global capacity and goal-specific buffers remain unchanged. Hindsight now has '
              'more eligible start states, so its update count can change; equal environment steps do not '
              'mean equal compute or updates. Options also alter primitive exploration under manager epsilon. '
              'Five seeds, one layout and in-distribution tasks do not establish general superiority.', '',
              '## Reproduce', '', '```powershell',
              'python compare_agents.py --config configs/comparison_initiation_sets.yaml',
              'python analyze_initiation_sets.py', '```', '',
              'Artifacts: learning_curves.png (all methods), learning_summary.png (adds goals and SPL), '
              'initiation_sets.png (seed-0 examples), raw CSVs, aggregate.json, saved option policies in '
              'options/, and transfer_probes.json. Seed 0 and the six most-selected final options were '
              'specified before inspecting results. Validation: 32 tests passed before training; 33 tests '
              'including an additional saved-policy probe check passed after training. Manager '
              'availability, replay from additional starts, goal context and baseline setting matching '
              'are checked. The analysis checks matched tasks and probes every final option.']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for kind in KINDS:
        print(kind,'SPL',fmt(kind,'evaluation_spl'),'late steps',fmt(kind,'final_training_steps'))
    env.close()


if __name__ == '__main__':
    main()
