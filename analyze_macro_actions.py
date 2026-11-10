"""Rebuild the three-environment action-grammar comparison and grammar artifacts."""

import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from analyze_random_goals import shortest_distances
from experiment import make_env


ROOT = Path(__file__).parent
OUTPUT = ROOT / 'results' / 'macro_actions'
SPECS = (
    ('four_rooms','comparison_macros_four_rooms','Four Rooms, fixed task'),
    ('nine_rooms','comparison_macros_nine_rooms','Nine Rooms, fixed task'),
    ('random_goals','comparison_macros_random_goals','Four Rooms, random start/goal'),
)
KINDS = ('q_learning','composed_pruned_hindsight','macro','macro_hindsight')
NAMES = {'q_learning':'Q-learning','composed_pruned_hindsight':'Cloned options + hindsight',
         'macro':'Macros, no macro replay','macro_hindsight':'Macros + hindsight action replay'}
ARROWS = ('↑','→','↓','←')


def read_csv(folder,name):
    with (folder/name).open(newline='',encoding='utf-8') as file:
        return list(csv.DictReader(file))


def flatten(macros,index,n_actions=4):
    result = []
    for symbol in macros[index]['rule']:
        assert symbol < n_actions+index, 'Grammar contains a forward/cyclic reference'
        result.extend([symbol] if symbol < n_actions else flatten(macros,symbol-n_actions,n_actions))
    return result


def symbol_label(action):
    return ARROWS[action] if action < 4 else f'M{action-4}'


def rollout(env,actions,start,task_goal):
    position = tuple(start)
    path = [position]
    collisions = 0
    for action in actions:
        dr,dc = env.MOVES[action]
        following = position[0]+dr,position[1]+dc
        if following in env.position_to_state:
            position = following
        else:
            collisions += 1
        path.append(position)
        if position == tuple(task_goal):
            break
    return dict(path=path,collisions=collisions,reached_task_goal=position == tuple(task_goal))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    figure = Figure(figsize=(18,13),layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(3,3)
    qualitative_figure = Figure(figsize=(13,15),layout='constrained')
    FigureCanvasAgg(qualitative_figure)
    qualitative_axes = qualitative_figure.subplots(3,2)
    all_results,qualitative,provenance = {},[],{}
    for row_index,(task,name,title) in enumerate(SPECS):
        folder = ROOT/'results'/name
        config = yaml.safe_load((folder/'config.yaml').read_text())
        summaries = read_csv(folder,'summary.csv')
        episodes = read_csv(folder,'episodes.csv')
        evaluation = read_csv(folder,'evaluation.csv')
        assert len(summaries) == 20
        assert len(evaluation) == 4*5*config['experiment']['evaluation_episodes']
        env = make_env(config['environment'])
        distances = shortest_distances(env)
        budget = config['experiment']['environment_steps']
        edges = np.linspace(0,budget,21)
        if env.observe_goal:
            for seed in range(5):
                histories = [[(r['start_cell'],r['goal_cell']) for r in episodes
                              if r['agent']==kind and int(r['seed'])==seed] for kind in KINDS]
                common = min(map(len,histories))
                assert all(history[:common] == histories[0][:common] for history in histories)
                panels = [[(r['start_cell'],r['goal_cell']) for r in evaluation
                           if r['agent']==kind and int(r['seed'])==seed] for kind in KINDS]
                assert all(panel==panels[0] for panel in panels)
        baseline_folder = ROOT/'results'/config['experiment']['reuse_baselines_from']
        provenance[task] = dict(baseline=str(baseline_folder),
            baseline_sha256={file:hashlib.sha256((baseline_folder/file).read_bytes()).hexdigest()
                             for file in ('config.yaml','summary.csv','episodes.csv','evaluation.csv')})
        results = {}
        for kind in KINDS:
            per_seed,curves = [],[]
            grammar_lines = [f'# {title}: {NAMES[kind]}', '',
                             'M indices are zero-based. Primitive actions: 0 up, 1 right, 2 down, 3 left.', '']
            for seed in range(5):
                summary = next(r for r in summaries if r['agent']==kind and int(r['seed'])==seed)
                history = [r for r in episodes if r['agent']==kind and int(r['seed'])==seed]
                assert int(history[-1]['environment_steps']) == budget
                complete = [r for r in history if not int(r['partial'])]
                test = [r for r in evaluation if r['agent']==kind and int(r['seed'])==seed]
                if env.observe_goal:
                    d = [distances[int(r['start_cell']),int(r['goal_cell'])] for r in test]
                else:
                    d = [distances[env.position_to_state[env.start],env.position_to_state[env.goal]]]*len(test)
                item = dict(seed=seed,
                    training_goals=sum(int(r['success']) for r in complete),
                    evaluation_spl=float(np.mean([int(r['success'])*dist/max(dist,int(r['steps'])) for r,dist in zip(test,d)])),
                    evaluation_optimal=float(np.mean([int(r['success']) and int(r['steps'])==dist for r,dist in zip(test,d)])),
                    **{key:float(summary[key]) for key in ('completed_episodes','final_training_success',
                       'final_training_steps','greedy_success','greedy_steps','hindsight_updates',
                       'longest_option_execution','max_option_depth','max_option_steps')})
                if kind.startswith('macro'):
                    saved = json.loads((folder/'macros'/f'{kind}_seed_{seed}.json').read_text())
                    assert saved['environment_steps'] == budget and saved['primitive_updates'] == budget
                    assert saved['hindsight_updates'] == int(summary['hindsight_updates'])
                    assert len(saved['macros']) == int(summary['options']) <= 20
                    assert saved['hindsight_updates'] == saved['online_hindsight_updates']+saved['discovery_hindsight_updates']
                    if kind=='macro':
                        assert saved['hindsight_updates'] == 0
                    for key in ('primitive_updates','manager_macro_updates','online_hindsight_updates','discovery_hindsight_updates'):
                        item[key] = saved[key]
                    macros = saved['macros']
                    grammar_lines += [f'## Seed {seed}', '',
                        '| Macro | Rule | Primitive expansion | Depth | Top-level selections | Observed lengths (steps: count) |',
                        '| --- | --- | --- | --- | --- | --- |']
                    for index,macro in enumerate(macros):
                        expanded = flatten(macros,index)
                        assert expanded == macro['primitive_actions'] and 2 <= len(expanded) <= 32
                        expected_depth = 1+max((macros[a-4]['depth'] for a in macro['rule'] if a>=4),default=0)
                        assert macro['depth'] == expected_depth
                        assert all(int(length) <= len(expanded) for length in macro['execution_lengths'])
                        rule = ' '.join(symbol_label(a) for a in macro['rule'])
                        expansion = ''.join(ARROWS[a] for a in expanded)
                        grammar_lines.append(f"| M{index} | {rule} | {expansion} | {macro['depth']} | "
                                             f"{macro['selections']} | {macro['execution_lengths']} |")
                    if kind=='macro_hindsight':
                        starts = {int(s) for macro in macros for s in macro['start_states']}
                        item['library_used_start_positions'] = len({s//len(env.positions) if env.observe_goal else s for s in starts})
                        item['library_used_task_goals'] = len({s%len(env.positions) for s in starts}) if env.observe_goal else 1
                        if seed == 0:
                            selected = [m for m in macros if m['selections']]
                            most_used = max(selected,key=lambda m:m['selections'])
                            longest_used = max(selected,key=lambda m:(max(map(int,m['execution_lengths']),default=0),m['selections']))
                            for column,macro in enumerate((most_used,longest_used)):
                                slot = macro['action']-4
                                starts = sorted(macro['start_states'],key=lambda s:-macro['start_states'][s])[:3]
                                examples = []
                                axis = qualitative_axes[row_index,column]
                                grid = np.zeros((env.size,env.size))
                                for r,c in env.walls:
                                    grid[r,c] = 1
                                axis.imshow(grid,cmap='Greys',vmin=0,vmax=1)
                                for number,state in enumerate(starts):
                                    state = int(state)
                                    cell = state//len(env.positions) if env.observe_goal else state
                                    goal = env.positions[state%len(env.positions)] if env.observe_goal else env.goal
                                    sample = rollout(env,macro['primitive_actions'],env.positions[cell],goal)
                                    examples.append(dict(start=env.positions[cell],task_goal=goal,**sample))
                                    points = np.array(sample['path'])
                                    axis.plot(points[:,1],points[:,0],marker='o',ms=3,lw=1.6,label=f'Start {number+1}; {sample["collisions"]} blocked')
                                    axis.scatter(points[0,1],points[0,0],marker='s',s=70)
                                rule = ' '.join(symbol_label(a) for a in macro['rule'])
                                expansion = ''.join(ARROWS[a] for a in macro['primitive_actions'])
                                axis.set_title(f"{title}: {'most selected' if column==0 else 'longest executed'}\n"
                                    f"M{slot} → {rule} | depth {macro['depth']} | {len(macro['primitive_actions'])} steps\n{expansion}",fontsize=10)
                                axis.set(xlabel='Column',ylabel='Row',xticks=range(env.size),yticks=range(env.size))
                                axis.tick_params(labelsize=7)
                                axis.legend(fontsize=7)
                                qualitative.append(dict(task=task,selection='most_selected' if column==0 else 'longest_executed',
                                    macro=macro,example_rollouts=examples))
                per_seed.append(item)
                ends = np.array([int(r['environment_steps']) for r in complete])
                success = np.array([int(r['success']) for r in complete])
                steps = np.array([int(r['steps']) for r in complete])
                curves.append([
                    [success[(ends>lo)&(ends<=hi)].mean() for lo,hi in zip(edges,edges[1:])],
                    [steps[(ends>lo)&(ends<=hi)].mean() for lo,hi in zip(edges,edges[1:])],
                    [success[ends<=hi].sum() for hi in edges[1:]],
                ])
            if kind.startswith('macro'):
                (folder/f'{kind}_grammar.md').write_text('\n'.join(grammar_lines)+'\n',encoding='utf-8')
            results[kind] = dict(per_seed=per_seed,metrics={key:dict(
                mean=float(np.mean([r[key] for r in per_seed])),sd=float(np.std([r[key] for r in per_seed],ddof=1)))
                for key in per_seed[0] if key != 'seed'})
            for axis,values in zip(axes[row_index],np.array(curves).transpose(1,0,2)):
                mean,sd = values.mean(axis=0),values.std(axis=0,ddof=1)
                line, = axis.plot(edges[1:],mean,label=NAMES[kind])
                axis.fill_between(edges[1:],mean-sd,mean+sd,alpha=.12,color=line.get_color())
            print(title,kind,'late steps',results[kind]['metrics']['final_training_steps']['mean'],
                  'evaluation',results[kind]['metrics']['greedy_steps']['mean'],flush=True)
        for axis,label in zip(axes[row_index],('Training success','Training steps/episode','Cumulative training goals')):
            axis.set(xlabel='Primitive environment steps',ylabel=label,title=title)
            axis.grid(alpha=.25)
        axes[row_index,0].set_ylim(0,1.05)
        axes[row_index,0].legend(fontsize=7)
        all_results[task] = results
        env.close()
    figure.suptitle('Recursive action grammars: matched environment budgets, five seeds, mean ± sample SD\n'
                    'Fixed tasks: 50 evaluation episodes/seed; random tasks: 1,000. Curves show training policies.')
    figure.savefig(OUTPUT/'learning_curves.png',dpi=150)
    qualitative_figure.suptitle('Seed 0, fixed selection rule: most-selected and longest-executed final HAR macros\n'
        'Same stored action sequence from up to three observed starts; squares mark starts. Walls can block moves.')
    qualitative_figure.savefig(OUTPUT/'macro_examples.png',dpi=150)
    (OUTPUT/'aggregate.json').write_text(json.dumps(all_results,indent=2),encoding='utf-8')
    (OUTPUT/'qualitative.json').write_text(json.dumps(qualitative,indent=2),encoding='utf-8')
    (OUTPUT/'runtime.json').write_text(json.dumps(dict(python=platform.python_version(),
        packages={name:importlib.metadata.version(name) for name in ('numpy','gymnasium','PyYAML','matplotlib')},
        baseline_provenance=provenance),indent=2),encoding='utf-8')

    def fmt(task,kind,key):
        value = all_results[task][kind]['metrics'][key]
        return f"{value['mean']:.3f} ± {value['sd']:.3f}"
    lines = ['# Recursive macro-actions with Hindsight Action Replay', '',
        '## Paper guidance and implementation', '',
        'Guidance: Christodoulou, Lange, Shafti and Faisal (2019), '
        '[Reinforcement Learning with Structured Hierarchical Grammar Representations of Actions](https://arxiv.org/abs/1910.02876). '
        'The paper introduces AG-RL and Hindsight Action Replay (HAR). This repo implements a tabular adaptation '
        'inspired by grammar-based action expansion and HAR, not a reproduction of its Atari agents.', '',
        'Our implementation stores binary rules over primitives or older macro symbols. It compresses each '
        'successful primitive trajectory with existing rules, counts repeated digrams, and adds one qualifying '
        'nonoverlapping pair every 10 episodes. Occurrence threshold is 3 and a new rule must save tokens after '
        'a two-token RHS cost. This is a greedy repeated-pair compressor, not exact Sequitur or the paper’s '
        'information-theoretic grammar criterion. Episodes are never concatenated for matching. Repeated '
        'primitive expansions are deduplicated. Rules reference only older rules, making recursion acyclic. '
        'A cached expansion supports matching, while runtime execution traverses the recursive rule stack.', '',
        'Macros are fixed open-loop action sequences available in every state and under every task goal. '
        'They execute to sequence completion, true terminal goal or time limit. They do not have a spatial '
        'subgoal, cloned state policy or unknown-state termination. A wall consumes an action and leaves '
        'position unchanged; global availability does not guarantee useful navigation.', '',
        'The tabular manager learns actual discounted duration returns with gamma**duration bootstrapping; '
        'only true terminal states remove the bootstrap. Primitive Q-values learn on every step, even inside '
        'macros. New macro columns copy the value of their first primitive, preserving existing columns. '
        'The HAR version additionally relabels exact observed primitive sequences as macro executions at '
        'their actual start observation and replays a new macro against up to 100 compatible segments from '
        'the latest 100 completed episodes. Goal-interrupted matching prefixes are valid; nonterminal '
        'incomplete prefixes are not relabelled. At most one longest compatible suffix per macro per step '
        'is used online. The actual selected-macro transition is not duplicated. Rewards and goals stay '
        'unchanged, and episode boundaries are never crossed.', '',
        'Differences from the paper: training successful trajectories replace separate elite greedy '
        'grammar rollouts; tabular online updates replace neural replay training; no action-balanced replay, '
        'Abandon Ship heuristic or macro exploration bonus. These choices keep the current exploration '
        'settings and environment budgets. Macro ablation disables macro relabelling/warm-start replay '
        'but retains primitive learning inside macros, value copying and the same induction procedure. '
        'Its learned grammar can differ because training trajectories differ. This is not an equal-update control.', '',
        '## Matched experiments', '',
        'Five seeds (0–4). Four Rooms fixed task: 100,000 steps and 50 evaluations/seed. Nine Rooms fixed '
        'task: 300,000 steps and 50 evaluations/seed. Four Rooms random distinct start/goal with both '
        'positions observed: 1,000,000 steps and 1,000 matched evaluations/seed. Alpha=.1, gamma=.99, '
        'epsilon=.2, buffers=100, discovery interval=10, minimum occurrences=3. Library cap=20, macro '
        'maximum length=32. Rules are appended without pruning and freeze once capacity is full. The '
        'cloned-option baseline retains its original pruning. Settings were fixed before running; no tuning '
        'or post-result budget extension.', '',
        'Existing Q-learning and cloned-option baselines are reused after exact config/variant/seed/budget '
        'validation. Raw data are included in each new comparison directory. Baseline file hashes and '
        'package versions are in runtime.json. Macro and ablation runs are new. Some jobs ran concurrently, '
        'so wall-clock times are not compared. Equal environment steps do not imply equal computation, '
        'memory or learning updates. HAR adds extra updates; availability and duration change exploration.', '',
        'Final evaluation freezes all learning and grammar induction on copied agents. Late training '
        'uses complete episodes ending in the final 20% of each budget; partial final episodes are excluded. '
        'Curves use 20 equal step bins, mean ± sample SD across trained seeds, and are training-policy curves. '
        'Random-task training prefixes and evaluation panels are checked for pairing; evaluation panels '
        'overlap across seeds under the existing seed formula. BFS distances are for analysis only.', '',
        '## Results', '',
        '| Task | Agent | Late-training steps | Training goals | Final success fraction | Final steps | Final SPL |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for task,_,title in SPECS:
        for kind in KINDS:
            lines.append('| '+title+' | '+NAMES[kind]+' | '+' | '.join(fmt(task,kind,key) for key in (
                'final_training_steps','training_goals','greedy_success','greedy_steps','evaluation_spl'))+' |')
    lines += ['', 'SPL is success * shortest_distance / max(shortest_distance, executed_steps). Failures '
              'score zero. Final step counts include failures at their time limit. Sample SD describes '
              'five-seed variability; no significance claim is made.', '', '### Measured comparisons', '']
    for task,_,title in SPECS:
        metrics = all_results[task]
        har = metrics['macro_hindsight']['metrics']
        no_har = metrics['macro']['metrics']
        q = metrics['q_learning']['metrics']
        options = metrics['composed_pruned_hindsight']['metrics']
        changes = [100*(har['final_training_steps']['mean']/other['final_training_steps']['mean']-1)
                   for other in (q,options,no_har)]
        lines += [f"- {title}: HAR late-training steps change {changes[0]:+.2f}% versus Q-learning, "
                  f"{changes[1]:+.2f}% versus cloned options, and {changes[2]:+.2f}% versus macros without macro replay. "
                  f"HAR reaches {har['training_goals']['mean']:.1f} training goals on average versus "
                  f"{q['training_goals']['mean']:.1f} for Q-learning. Final HAR success is "
                  f"{100*har['greedy_success']['mean']:.2f}% and mean path length {har['greedy_steps']['mean']:.3f}. "
                  f"Its final SPL changes {har['evaluation_spl']['mean']-q['evaluation_spl']['mean']:+.4f} versus Q-learning."]
    lines += ['', 'Training-policy efficiency and frozen final path efficiency are different outcomes. '
              'A long macro can reduce manager decisions and still take detours. HAR improves data reuse '
              'but adds many more updates; no equal-update ordinary-replay control is included.', '',
              'Random-task final HAR success by seed is '
              f"{[round(100*r['greedy_success'],1) for r in all_results['random_goals']['macro_hindsight']['per_seed']]} percent. "
              'This substantial seed variability is hidden by a single average. Fixed-goal final HAR paths '
              'are 16 steps in every Four Rooms seed; Nine Rooms has 30-step paths in seeds 2 and 4, '
              'while Q-learning has optimal 28-step paths in every seed. HAR improves average late-training '
              'episode length over the macro ablation in all three tasks, but does not remove the '
              'random-task failures or establish a general performance advantage.', '',
              '## Actual macro use and replay', '',
              '| Task | Agent | Longest observed execution per seed | Stored maximum depth per seed | Online macro HAR updates | Discovery HAR updates |',
              '| --- | --- | --- | --- | --- | --- |']
    for task,_,title in SPECS:
        for kind in KINDS[2:]:
            rows = all_results[task][kind]['per_seed']
            lines.append('| '+title+' | '+NAMES[kind]+' | '+str([int(r['longest_option_execution']) for r in rows])+
                ' | '+str([int(r['max_option_depth']) for r in rows])+' | '+fmt(task,kind,'online_hindsight_updates')+
                ' | '+fmt(task,kind,'discovery_hindsight_updates')+' |')
    lines += ['', 'Unlike the earlier cloned policies, macro definitions do not contain absolute position '
              'or task-goal keys. The manager still uses the full observed state, including the goal when '
              'present, to choose a useful action. This shares the action sequence, not the manager values '
              'or a goal-reaching policy across states.', '',
              '| Task | Physical starting cells where some HAR macro was selected, per seed | Task goals where some HAR macro was selected, per seed |',
              '| --- | --- | --- |']
    for task,_,title in SPECS:
        rows = all_results[task]['macro_hindsight']['per_seed']
        lines.append('| '+title+' | '+str([r['library_used_start_positions'] for r in rows])+' | '+
                     str([r['library_used_task_goals'] for r in rows])+' |')
    lines += ['', '## Qualitative examples', '',
              'Seed 0 is fixed for illustration. For each environment, show the most-selected final HAR '
              'macro and the one with the longest observed execution (ties by selections). Draw the same '
              'stored primitive expansion from up to three most-used observed starts. These are deterministic '
              'saved-sequence rollouts in the grid, not additional training. Path overlap can hide repeated '
              'wall collisions; legends report blocked moves. Full grammars, expansions, depths and '
              'observed lengths for every seed are in each comparison’s macro*_grammar.md and macros/*.json.', '',
              '| Task | Selection rule | Rule | Expanded primitive actions | Depth | Top-level selections |',
              '| --- | --- | --- | --- | --- | --- |']
    for example in qualitative:
        macro = example['macro']
        lines.append('| '+example['task']+' | '+example['selection']+' | M'+str(macro['action']-4)+' → '+
            ' '.join(symbol_label(a) for a in macro['rule'])+' | '+''.join(ARROWS[a] for a in macro['primitive_actions'])+
            f" | {macro['depth']} | {macro['selections']} |")
    lines += ['', '## Limits and reproduction', '',
              'A sequence can be executable everywhere and still hit walls or take unnecessary detours. The '
              'current implementation has no feedback-based abort except episode termination. A library '
              'filled early can preserve poor rules; we did not add pruning or tune capacity afterward. '
              'HAR can correct manager values but cannot change a stored sequence. Five seeds and these '
              'layouts do not establish generalization to unseen maps or replicate the paper’s Atari results.', '',
              '```powershell',
              'python compare_agents.py --config configs/comparison_macros_four_rooms.yaml',
              'python compare_agents.py --config configs/comparison_macros_nine_rooms.yaml',
              'python compare_agents.py --config configs/comparison_macros_random_goals.yaml',
              'python analyze_macro_actions.py', '```', '',
              'Remove experiment.reuse_baselines_from to retrain baselines. All original result folders '
              'are preserved. Validation: 44 tests passed; recursive expansions, acyclic grammar references, '
              'length limits, snapshot counters, exact budgets, and paired random tasks are checked. '
              'Frozen evaluation does not learn or induce rules. Every benchmark macro library is saved before evaluation.']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved combined report, learning curves and macro examples to',OUTPUT)


if __name__ == '__main__':
    main()
