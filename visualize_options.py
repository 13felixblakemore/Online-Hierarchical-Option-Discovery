"""Train and inspect options: python visualize_options.py --seed 0.

Creates a standalone HTML viewer, PNG overview, and raw JSON snapshots.
Snapshot collection never samples actions or changes the agent's RNG.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from experiment import AGENT_LABELS, make_agent, make_env, update_agent, variant_settings


def snapshot(agent, step):
    return dict(step=step, episodes=agent.episodes,
                pruned=sum(len(e['option_ids']) for e in agent.pruning_history),
                hindsight_updates=agent.hindsight_updates,
                longest_option_execution=agent.longest_option_execution, options=[
        dict(id=o.option_id, start=o.start_state, goal=o.goal_state,
             max_steps=o.max_steps, max_decisions=o.max_decisions,
             created_step=o.created_step, selections=o.selections,
             last_used_step=o.last_used_step,
             hindsight_updates=getattr(o, 'hindsight_updates', 0),
             depth=o.depth, children=[agent.options[child].option_id for child in o.children],
             execution_lengths=dict(o.execution_lengths),
             manager_value=float(agent.q[o.start_state, agent.n_actions+i]),
             action_selection=o.action_selection, temperature=o.temperature,
             initiation_mode=o.initiation_mode, initiation_states=sorted(o.initiation_states),
             initiation_values={str(s): float(agent.q[s, agent.n_actions+i])
                                for s in sorted(o.initiation_states)},
             action_probabilities={str(s): o.action_probabilities(s) for s in o.action_counts},
             policy={str(s): dict(counts) for s, counts in o.action_counts.items()})
        for i, o in enumerate(agent.options)
    ])


def save_overview(data, path):
    figure = Figure(figsize=(16, 8), layout='constrained')
    FigureCanvasAgg(figure)
    stages = data['snapshots']
    axes = np.atleast_1d(figure.subplots(2, (len(stages)+1)//2)).ravel()
    positions = data['positions']
    for axis, stage in zip(axes, stages):
        grid = np.zeros((data['size'], data['size']))
        for row, col in data['walls']:
            grid[row, col] = np.nan
        for option in stage['options']:
            row, col = positions[option['start']]
            grid[row, col] += 1
        cmap = colormaps['YlOrRd'].copy()
        cmap.set_bad('#333333')
        axis.imshow(grid, cmap=cmap, vmin=0, vmax=max(1, max(
            sum(o['start'] == s for o in stages[-1]['options'])
            for s in range(len(positions)))))
        for option in stage['options']:
            row, col = positions[option['start']]
            target_row, target_col = positions[option['goal']]
            axis.annotate('', xy=(target_col, target_row), xytext=(col, row),
                          arrowprops=dict(arrowstyle='->', color='#2468af',
                                          alpha=0.6, lw=1, linestyle='--'))
        start = data.get('task_start', data['config']['environment'].get('start', [1, 1]))
        if not data['config']['environment'].get('random_start', False):
            axis.scatter(start[1], start[0], marker='o', s=180,
                         facecolor='#159653', edgecolor='white', linewidth=1.5, zorder=5)
            axis.text(start[1], start[0], 'S', color='white', ha='center',
                      va='center', fontsize=9, weight='bold', zorder=6)
        goal = data['task_goal']
        axis.scatter(goal[1], goal[0], marker='*', s=260,
                     facecolor='#8b36c5', edgecolor='white', linewidth=1, zorder=5)
        axis.text(goal[1]+0.5, goal[0], 'G', color='#6a219c',
                  va='center', fontsize=10, weight='bold', zorder=6)
        longest = max((int(length) for o in stage['options']
                       for length in o.get('execution_lengths', {})), default=0)
        depth = max((o.get('depth', 1) for o in stage['options']), default=0)
        axis.set_title(f"{stage['step']:,} steps | {len(stage['options'])} options | depth {depth}\nPruned: {stage.get('pruned', 0)} | Longest active: {longest} primitive steps", fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes[len(stages):]:
        axis.set_visible(False)
    figure.suptitle('Learned options over training\n'
                   'Warm cells: number of options starting there. Dashed arrows: start to target, not executed paths.')
    start_label = ('Random start' if data['config']['environment'].get('random_start', False)
                   else 'S: task start (row, col) = ' + str(tuple(start)))
    figure.legend(handles=[
        Line2D([], [], marker='o', color='#159653', linestyle='', label=start_label),
        Line2D([], [], marker='*', color='#8b36c5', linestyle='',
               label='G: task goal (row, col) = ' + str(tuple(goal))),
    ], loc='outside lower center', ncol=2)
    figure.savefig(path, dpi=150)


def representative_rollout(data, stage, slot, rng=None):
    """Execute saved policies; use modal paths unless an analysis RNG is supplied."""
    options = stage['options']
    option = options[slot]
    state = option['start']
    path = [data['positions'][state]]
    lookup = {tuple(position): i for i, position in enumerate(data['positions'])}
    stack = [dict(slot=slot, steps=0, decisions=0)]
    actions = []
    while stack:
        frame = stack[-1]
        current = options[frame['slot']]
        counts = current['policy'][str(state)]
        best = max(counts.values())
        if rng is None:
            action = min(int(a) for a, count in counts.items() if count == best)
        else:
            probabilities = current['action_probabilities'][str(state)]
            action = int(rng.choice([int(a) for a in probabilities],
                                    p=list(probabilities.values())))
        frame['decisions'] += 1
        if action >= data['n_actions']:
            stack.append(dict(slot=action-data['n_actions'], steps=0, decisions=0))
            continue
        row, col = data['positions'][state]
        dr, dc = data['moves'].get(str(action), data['moves'].get(action))
        state = lookup.get((row+dr, col+dc), state)
        path.append(data['positions'][state])
        actions.append(action)
        for frame in stack:
            frame['steps'] += 1
        for i, frame in enumerate(stack):
            current = options[frame['slot']]
            if state == current['goal'] or frame['steps'] >= current['max_steps']:
                del stack[i:]
                break
        while stack:
            frame = stack[-1]
            current = options[frame['slot']]
            if (frame['decisions'] >= current.get('max_decisions', 2)
                    or str(state) not in current['policy']):
                stack.pop()
            else:
                break
        if data['positions'][state] == data['task_goal']:
            break
    return dict(option_id=option['id'], positions=path, actions=actions,
                reached_target=state == option['goal'])


def save_option_details(data, path):
    """Show actual primitive movement of the six longest-used final options."""
    stage = data['snapshots'][-1]
    ranked = sorted(enumerate(stage['options']), key=lambda pair: (
        max(map(int, pair[1].get('execution_lengths', {})), default=0),
        pair[1]['selections']), reverse=True)[:6]
    figure = Figure(figsize=(15, 10), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3).ravel()
    grid = np.zeros((data['size'], data['size']))
    for row, col in data['walls']:
        grid[row, col] = 1
    rollouts = []
    for axis, (slot, option) in zip(axes, ranked):
        rollout = representative_rollout(data, stage, slot)
        rollouts.append(rollout)
        axis.imshow(grid, cmap='Greys', vmin=0, vmax=1)
        points = rollout['positions']
        for step, (first, second) in enumerate(zip(points, points[1:]), 1):
            row, col = first
            tr, tc = second
            axis.annotate('', xy=(tc, tr), xytext=(col, row),
                          arrowprops=dict(arrowstyle='->', color='#2468af', lw=2))
            axis.text((col+tc)/2+0.12, (row+tr)/2-0.12, str(step),
                      fontsize=8, color='#123b77')
        start = data['positions'][option['start']]
        target = data['positions'][option['goal']]
        axis.scatter(start[1], start[0], color='#159653', s=75, zorder=5)
        axis.scatter(target[1], target[0], marker='D', color='#e58712', s=60, zorder=5)
        axis.set_title(f"Option {option['id']} | {len(rollout['actions'])} primitive steps | depth {option['depth']}\n"
                       f"{tuple(start)} to {tuple(target)} | child options: {option['children']}", fontsize=10)
        axis.set_xticks(range(data['size']))
        axis.set_yticks(range(data['size']))
        axis.tick_params(labelsize=7)
        axis.set(xlabel='Column', ylabel='Row')
    for axis in axes[len(ranked):]:
        axis.set_visible(False)
    figure.suptitle('Final learned option policies: representative primitive paths\n'
                   'Green: initiation. Orange: target. Numbers: primitive step order. Modal ties choose the lowest action ID.')
    figure.savefig(path, dpi=150)
    return rollouts


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Learned options over training</title>
<style>
body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#203047;background:#f6f8fb}
.layout{display:flex;gap:28px;flex-wrap:wrap}.card{background:white;border-radius:12px;padding:20px;box-shadow:0 2px 10px #0001}
svg{width:min(90vw,550px);height:auto}select{font:inherit;padding:6px;width:100%}input{width:100%}
#detail{max-width:380px;line-height:1.6}h1{font-size:26px}label{display:block;margin:15px 0}small{color:#536078}
</style>
<h1>How the agent's options develop</h1>
<p id="caption"></p><label>Training stage <input id="stage" type="range" min="0" step="1"></label>
<div class="layout"><div class="card"><svg id="grid" role="img" aria-label="Room environment option policy"></svg></div>
<div class="card" id="detail"><label>Option <select id="option"></select></label><div id="stats"></div>
<p><b>Green circle:</b> initiation state. <b>Orange diamond:</b> target / termination state.
<b>Blue arrows:</b> actions supported by the option policy; child-option calls point to the child's target.
Arrow labels show selection probabilities (demonstration fractions in older snapshots).</p>
<small>These are closed-loop cloned policies, not fixed action sequences. Options also stop at unseen states or their step limit.
Counts describe demonstrations used when the option was created. Policies are frozen in this implementation; manager values and selection counts can change.</small></div></div>
<script>
const data=__DATA__;
document.querySelector('#grid').setAttribute('viewBox',`-20 -20 ${data.size*40+40} ${data.size*40+40}`);
const slider=document.querySelector('#stage'), select=document.querySelector('#option');
slider.max=data.snapshots.length-1;
const pos=s=>data.positions[s];
function draw(){
 const stage=data.snapshots[+slider.value], o=stage.options.find(o=>o.id===+select.value);
 document.querySelector('#caption').textContent=`Seed ${data.seed} | ${stage.step.toLocaleString()} environment steps | ${stage.episodes} completed episodes | ${stage.options.length} active options | ${stage.pruned || 0} pruned`;
 let svg='<defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#2468af"/></marker></defs>';
 const walls=new Set(data.walls.map(p=>p.join(',')));
 for(let r=0;r<data.size;r++)for(let c=0;c<data.size;c++){
   svg+=`<rect x="${c*40}" y="${r*40}" width="40" height="40" fill="${walls.has([r,c].join(','))?'#333':'#fff'}" stroke="#ddd"/>`;
 }
 const [gr,gc]=data.task_goal; svg+=`<text x="${gc*40+20}" y="${gr*40+25}" text-anchor="middle" fill="#888">G</text>`;
 if(!data.config.environment.random_start){const [sr,sc]=data.task_start || data.config.environment.start || [1,1];svg+=`<text x="${sc*40+12}" y="${sr*40+12}" font-size="11" fill="#159653">S</text>`;}
 if(o){
  for(const [s,counts] of Object.entries(o.policy)){
   const [r,c]=pos(+s), best=Math.max(...Object.values(counts)), total=Object.values(counts).reduce((a,b)=>a+b,0);
   const probabilities=o.action_probabilities?.[s];
   for(const [a,n] of Object.entries(counts))if(probabilities ? probabilities[a]>0 : n===best){
    let dr,dc;
    if(+a<(data.n_actions || 4)){[dr,dc]=data.moves[a]}else{const child=stage.options[+a-(data.n_actions || 4)];const [tr,tc]=pos(child.goal);dr=tr-r;dc=tc-c;}
    const x=c*40+20,y=r*40+20;
    svg+=`<line x1="${x}" y1="${y}" x2="${x+dc*26}" y2="${y+dr*26}" stroke="#2468af" stroke-width="2" marker-end="url(#arrow)"/>`;
    svg+=`<text x="${x+dc*17+5}" y="${y+dr*17-5}" font-size="9" fill="#2468af">${+a>=data.n_actions?'O'+stage.options[+a-data.n_actions].id+' ':''}${Math.round((probabilities ? probabilities[a] : n/total)*100)}%</text>`;
   }
  }
  const [r,c]=pos(o.start),[tr,tc]=pos(o.goal);
  svg+=`<circle cx="${c*40+20}" cy="${r*40+20}" r="12" fill="none" stroke="#159653" stroke-width="3"/>`;
  svg+=`<path d="M${tc*40+20},${tr*40+8} l12,12 l-12,12 l-12,-12 Z" fill="none" stroke="#e58712" stroke-width="3"/>`;
  document.querySelector('#stats').innerHTML=`<h2>Option ${o.id}</h2>Start: (${r}, ${c})<br>Target: (${tr}, ${tc})<br>Created at step ${o.created_step.toLocaleString()}<br>Primitive-step limit: ${o.max_steps}<br>Hierarchy depth: ${o.depth || 1}<br>Child options: ${(o.children || []).join(', ') || 'none'}<br>Observed lengths (steps: count): ${JSON.stringify(o.execution_lengths || {})}<br>Selected ${o.selections} times since creation<br>Manager Q-value at start: ${o.manager_value.toFixed(4)}`;
 }else document.querySelector('#stats').textContent='No options discovered yet.';
 document.querySelector('#grid').innerHTML=svg;
}
function changeStage(){const old=select.value;select.replaceChildren();for(const o of data.snapshots[+slider.value].options){const el=document.createElement('option');el.value=o.id;el.textContent=`Option ${o.id}: (${pos(o.start)}) → (${pos(o.goal)})`;select.append(el)}if([...select.options].some(x=>x.value===old))select.value=old;draw()}
slider.oninput=changeStage;select.onchange=draw;changeStage();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parent
    parser.add_argument('--config', type=Path, default=root/'configs'/'comparison.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--agent', choices=[kind for kind in AGENT_LABELS
                                          if kind not in ('q_learning','macro','macro_hindsight')])
    parser.add_argument('--checkpoints', nargs='+', type=int,
                        help='Primitive steps at which to save snapshots')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    checkpoints = sorted(set([0] + (args.checkpoints or config.get('experiment', {}).get(
        'snapshot_steps', [0, 5000, 10000, 15000, 20000, 30000, 50000, 100000]))))
    if min(checkpoints) < 0:
        parser.error('Checkpoints must be nonnegative')
    if args.agent:
        config['agent'] = variant_settings(config['agent'], args.agent)
    mode = 'composed' if config['agent'].get('compose_options', False) else 'flat'
    if config['agent'].get('prune_options', False):
        mode += '_pruned'
    if config['agent'].get('manager_hindsight', False):
        mode += '_hindsight'
    if config['agent'].get('option_action_selection', 'most_frequent') == 'softmax':
        mode += f"_softmax_T{config['agent'].get('option_temperature', 1.0):g}"
    if config['agent'].get('option_initiation', 'single_state') != 'single_state':
        mode += '_support'
    snapshot_root = root/'results'/'option_snapshots'
    if config['environment'].get('type', 'four_rooms') != 'four_rooms':
        snapshot_root /= config['environment']['type']
    output = snapshot_root/mode/f'seed_{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    env = make_env(config['environment'])
    agent = make_agent(dict(config['agent'], type='repetition'), env, args.seed)
    stages = [snapshot(agent, 0)]
    state, _ = env.reset(seed=args.seed)
    try:
        for step in range(1, max(checkpoints)+1):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            update_agent(agent, state, action, reward, next_state, terminated, truncated)
            state = next_state
            if terminated or truncated:
                state, _ = env.reset()
            if step in checkpoints:
                stages.append(snapshot(agent, step))
                print(f'Step {step:,}: {len(agent.options)} options', flush=True)
    finally:
        env.close()
    data = dict(seed=args.seed, config=config, positions=env.positions, n_actions=agent.n_actions,
                walls=sorted(env.walls), size=env.size, moves=env.MOVES,
                task_start=env.start, task_goal=env.goal, snapshots=stages,
                pruning_history=agent.pruning_history)
    serialized = json.dumps(data)
    (output/'snapshots.json').write_text(serialized, encoding='utf-8')
    (output/'options.html').write_text(HTML.replace('__DATA__', serialized), encoding='utf-8')
    save_overview(data, output/'overview.png')
    rollouts = save_option_details(data, output/'option_details.png')
    (output/'representative_rollouts.json').write_text(json.dumps(rollouts), encoding='utf-8')
    print(f'Saved viewer and snapshots to {output}', flush=True)


if __name__ == '__main__':
    main()
