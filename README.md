# Online Hierarchical Skill Discovery from Repetition

A small reinforcement learning research repo exploring whether repeated successful
behaviour can become reusable, temporally extended actions (**options**). It compares
tabular Q-learning with online option discovery, recursive option composition,
inactivity pruning, and retrospective manager updates in Four Rooms and Nine Rooms.

The saved experiments show improvements in some training metrics on Four Rooms.
The larger Nine Rooms task gives mixed results: the full method has shorter
late-training episodes, while plain Q-learning reaches more goals over the full
budget and has better final greedy paths. This is an experimental prototype with
five seeds per comparison, rather than evidence of a general advantage.

## Setup

Validated with Python 3.12. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -v
```

On Linux/macOS, activate with `source .venv/bin/activate`. Dependencies are Gymnasium,
NumPy, PyYAML, and Matplotlib. Plots use a noninteractive backend; no display server
is required. Dependency ranges in `requirements.txt` are compatibility bounds,
not a lockfile for the historical runs.

## Repository guide

| Path | Purpose |
| --- | --- |
| `agents.py` | Random baseline, tabular Q-learning, repeated state-pair discovery, cloned option policies, and the semi-Markov manager |
| `macro_actions.py` | Recursive binary action grammar, globally available macro execution, and hindsight action replay |
| `four_rooms.py` | Gymnasium environment and shared room-grid dynamics |
| `nine_rooms.py` | Nine Rooms layout using the same dynamics |
| `experiment.py` | Shared environment/agent factories, named variants, transition updates, and CSV output |
| `train.py` | Train one agent for a fixed number of episodes |
| `compare_agents.py` | Compare methods under equal primitive environment-step budgets and evaluate final greedy policies |
| `visualize_options.py` | Rerun training and export option snapshots, an HTML viewer, and PNG figures |
| `analyze_softmax.py` | Rebuild softmax comparison plots, sampled option statistics, and the experiment report |
| `analyze_random_goals.py` | Analyze the random-start/random-goal comparison with shortest-path metrics |
| `analyze_initiation_sets.py` | Compare wider initiation sets and probe policy transfer from every physical cell |
| `analyze_macro_actions.py` | Compare recursive macros across all three tasks and export learned grammars and example paths |
| `configs/` | Single-agent settings and comparison protocols |
| `tests/` | Environment, option execution, pruning, hindsight, and experiment integration checks |
| `results/` | Recorded CSVs, exact run configs, reports, plots, and option snapshots |

## Environments

States are integer IDs of walkable cells. Coordinates in configs and figures are
zero-based `(row, column)`. Actions are `0=up`, `1=right`, `2=down`, `3=left`.
Movement is deterministic; hitting a wall or grid boundary leaves the state
unchanged. Reaching the goal gives reward 1 and terminates the episode; every
other step gives reward 0. Time limits truncate episodes and still bootstrap
learning targets. The original comparisons use fixed endpoints; the random-task
comparison resamples both endpoints each episode and exposes the goal in observations.

| Environment | Grid | Walkable states | Doorways | Start → goal | Shortest path | Episode limit |
| --- | --- | --- | --- | --- | --- | --- |
| Four Rooms | 11 × 11 | 104 | 4 | (1, 1) → (9, 9) | 16 steps | 200 steps |
| Nine Rooms | 17 × 17 | 237 | 12 | (1, 1) → (15, 15) | 28 steps | 500 steps |

Four Rooms also supports random tasks with `random_start: true`,
`random_goal: true`, and `observe_goal: true`. Each reset samples a goal uniformly
from the 104 walkable cells and a distinct start uniformly from the remaining 103.
The goal stays fixed during that episode. Both positions are encoded in the
tabular observation as `position_cell * 104 + goal_cell`, giving 10,816 states.
`divmod(observation, 104)` recovers the cell IDs; `env.decode_observation(state)`
returns their `(row, column)` coordinates. Random goals require goal observation.
The default fixed-goal environment retains its original position-only IDs.

Option initiation can be configured with `agent.option_initiation`:

- `single_state` (default): available only at the original demonstrated start.
- `policy_support`: available at every demonstrated policy state except its target.

The broader set permits starting partway through demonstrated behaviour. It does
not define actions at unseen states or guarantee that every supported start reaches
the target before its execution limit. Both manager availability and hindsight
use the same initiation set; hindsight updates the value at the actual replay start.
Task-goal context is still part of policy keys in the random-goal environment.

The [initiation-set experiment](results/comparison_initiation_sets/report.md)
uses the same five-seed random-task protocol. Broader sets averaged about two
states per final option. Sampled target-reaching success was 100% within those
sets, while forcing starts throughout the map encountered an undefined policy
at about 98% of cells. Final task success was tied at 100%; wider initiation did
not produce sustained deeper options. Its config reuses the validated previous
baselines; remove `experiment.reuse_baselines_from` to retrain them.

Options discovered on joint observations are specific to the episode's task goal.
The existing 20-option capacity is shared globally across goals; the algorithm
does not gain cross-goal skill transfer merely by observing the goal.

## How option discovery works

1. The manager chooses a primitive action or an option available at the current
   state, using epsilon-greedy exploration and random tie breaking.
2. Successful trajectories enter a bounded buffer. Every discovery interval,
   adjacent pairs of transitions are grouped by their start and end states.
   Self-returning pairs are excluded. Candidates are ranked by frequency, require
   a minimum occurrence count, and at most one unseen endpoint pair is added.
3. Each option clones state/action counts from its demonstrations. It chooses a
   most frequent action at each state, breaking modal ties randomly. It stops at
   its target, an unseen state, or its execution limit. Its policy stays fixed.
4. The manager learns with a semi-Markov target: discounted primitive rewards
   accumulated during execution, plus `gamma ** duration` times the next available
   manager value. Only true goal termination removes the bootstrap.

Flat discovery uses primitive transitions, including steps taken inside options.
Recursive discovery uses manager decisions, so a learned policy can call older
options. References to older options prevent cycles. Parents can interrupt a child
at their own target or step limit. Stored hierarchy depth and safety limits do not
measure the duration actually executed; observed lengths are recorded separately.

Optional **pruning** removes options unused for a configured number of primitive
steps. It runs at episode boundaries on discovery intervals, gives new options a
grace period, and protects dependencies of recently used options. Surviving Q-values,
child calls, and discovery replay are remapped together; replay is split at removed
calls so unrelated transitions are never joined. Stable option IDs are not reused.

Optional **manager hindsight** warm-starts a new option's manager value from a
separate successful primitive-trajectory buffer. Segments must agree with modal
policy actions, including nested calls, through actual option termination. It uses
the original rewards and goals, adds at most 100 replay updates per new option in
the recorded runs, and does not count replay as real usage or environment steps.

## Train an agent

```powershell
python train.py --config configs/config.yaml
python train.py --config configs/config_repetition.yaml
python train.py --config configs/config_nine_rooms.yaml
```

These run Q-learning in Four Rooms (2,000 episodes), flat repetition in Four Rooms
(2,000 episodes), and recursive repetition with pruning and hindsight in Nine Rooms
(5,000 episodes), respectively. `agent.type` accepts `random`, `q_learning`, or
`repetition`; the option features are switches on the repetition agent.

`training` controls episodes and logging. `output.csv` and `output.q_table` resolve
relative to the config directory. Learning curves are written to
`results/<config-stem>_learning_curves.png`; their trailing window is in episodes.
Q-table export contains only the numeric manager table, so it is not a complete
checkpoint for an option agent: option policies and execution state are needed too.

Option policies use `option_action_selection: most_frequent` by default, preserving
random tie breaking. To sample a softmax over demonstrated action counts, add:

```yaml
agent:
  type: repetition
  option_action_selection: softmax
  option_temperature: 1.0
```

At each state, `P(action) = exp(count(action) / temperature) / sum(exp(count / temperature))`
over actions demonstrated there, including child-option calls. Counts are raw
occurrence counts, not normalized proportions. Lower temperatures concentrate on
the most frequent actions; higher temperatures approach uniform sampling among
demonstrated actions. Temperature must be finite and positive. The manager's
epsilon-greedy selection and option termination rules stay the same; greedy
manager evaluation still samples the option's internal policy.

With softmax, hindsight accepts any action with nonzero policy probability,
including lower-count actions. Replay remains based on policy support and is not
importance weighted, so it does not estimate the softmax policy's expected return
without bias. Snapshot viewers display the configured action probabilities;
representative path figures continue to show a most-probable path. The original
experiments used the default most-frequent policy; `comparison_softmax.yaml`
adds matched softmax comparisons at T=1.

## Recorded experiments

### Recursive macro-actions

`MacroActionAgent` in `macro_actions.py` is a separate, tabular adaptation guided
by [Action Grammar RL](https://arxiv.org/abs/1910.02876), rather than a reproduction
of the paper's Atari agents. It stores recursive binary rules such as
`M1 → M0 M0`, with older rules as children and cached primitive expansions.
Macros are fixed action sequences available at every state and task goal; the
manager learns their state-dependent values. Walls can block their moves, and
they may take detours. Reaching the task goal interrupts execution immediately.

Grammar induction compresses successful trajectories using existing rules, then
adds one frequent nonoverlapping pair every discovery interval. It does not join
episodes. The default library cap is 20 and maximum primitive length is 32, with
no macro pruning. This is a greedy repeated-pair compressor, not exact Sequitur.
New macro columns copy their first primitive's values. Primitive Q-values learn
inside macros; with hindsight enabled, matching observed action sequences also
update macro values online and a new macro receives bounded retrospective replay.
Original rewards and task goals are preserved. Evaluation freezes learning and
grammar induction.

For single-agent training, use `agent.type: macro` and
`agent.manager_hindsight: true`. Macro-specific `macro_max_length` defaults to 32;
the common learning, discovery, buffer and capacity settings remain configurable.
A Q-table alone is not a full macro checkpoint: the grammar is required too.

```powershell
python compare_agents.py --config configs/comparison_macros_four_rooms.yaml
python compare_agents.py --config configs/comparison_macros_nine_rooms.yaml
python compare_agents.py --config configs/comparison_macros_random_goals.yaml
python analyze_macro_actions.py
```

These compare Q-learning, the cloned-option full method, macros without macro
replay, and macros with replay. Existing baselines are reused only after settings
and budget validation; remove `experiment.reuse_baselines_from` to rerun them.
Each comparison saves raw CSVs and every learned macro library. The combined
[macro report](results/macro_actions/report.md), curves, qualitative paths and
runtime provenance are in `results/macro_actions/`; per-seed rule listings are
saved as `macro_grammar.md` and `macro_hindsight_grammar.md` in each comparison.

The five-seed results show average late-training episode lengths of 19.68
(Four Rooms) and 34.50 (Nine Rooms) for hindsight macros, compared with Q-learning's
21.76 and 37.76. Final fixed-task success is 100%, with all Four Rooms paths
optimal and two Nine Rooms seeds retaining 30-step paths. Random-task success
averages 93.42%, below both baselines at 100%, with substantial seed variation.
The HAR agent executes macros up to 30 steps long and stores depths up to 8.
Longer hierarchies and global executability do not guarantee efficient navigation.

### Protocols

All comparisons use paired seeds 0–4, `alpha=0.1`, `gamma=0.99`, and manager
`epsilon=0.2`. Options have 20 active slots, discovery every 10 episodes, a minimum
of 3 occurrences, and a 100-successful-episode buffer. Pruning uses a 10,000-step
inactivity threshold. No per-method hyperparameter tuning was reported.

| Experiment config | Methods compared | Steps per method/seed | Saved report |
| --- | --- | --- | --- |
| `configs/comparison.yaml` | Q-learning vs flat two-step options | 100,000 | [Initial comparison](results/comparison/report.md) |
| `configs/comparison_recursive.yaml` | Adds recursive composition | 100,000 | [Recursive comparison](results/comparison_recursive/report.md) |
| `configs/comparison_pruning.yaml` | Q-learning, flat/recursive options, each with/without pruning | 100,000 | [Pruning comparison](results/comparison_pruning/report.md) |
| `configs/comparison_hindsight.yaml` | Recursive + pruning, hindsight off/on | 100,000 | [Hindsight comparison](results/comparison_hindsight/report.md) |
| `configs/comparison_nine_rooms.yaml` | All six methods on Nine Rooms | 300,000 | [Nine Rooms comparison](results/comparison_nine_rooms/report.md) |
| `configs/comparison_softmax.yaml` | Six original methods plus flat/full softmax, T=1, on Four Rooms | 100,000 | [Softmax comparison](results/comparison_softmax/report.md) |
| `configs/comparison_random_goals.yaml` | Q-learning vs composed/pruned/hindsight, random start and goal with both observed | 1,000,000 | [Random-task comparison](results/comparison_random_goals/report.md) |
| `configs/comparison_initiation_sets.yaml` | Adds demonstrated-policy initiation sets to the random-task comparison | 1,000,000 | [Initiation-set comparison](results/comparison_initiation_sets/report.md) |

The random-task experiment uses 1,000 matched evaluation tasks per seed, versus
50 evaluation episodes in the fixed-goal comparisons. Both agents reach 100%
final success; mean evaluation path lengths are 9.074 and 9.067 steps for
Q-learning and composed/pruned/hindsight respectively. Their learning curves
are close. The option agent's longest execution is two steps in every seed,
and all final options have depth 1. See its report for shortest-path metrics,
goal-specific option semantics, and the limitations of global option capacity.

Run any comparison from the repository root, for example:

```powershell
python compare_agents.py --config configs/comparison_hindsight.yaml
python compare_agents.py --config configs/comparison_nine_rooms.yaml
```

### Four Rooms findings

Late-training steps per episode, mean ± sample standard deviation across seeds
(lower is better):

| Method | Steps per episode |
| --- | --- |
| Q-learning | 21.758 ± 0.494 |
| Flat two-step options | 20.418 ± 0.403 |
| Flat + pruning | 20.264 ± 0.421 |
| Recursive | 20.544 ± 0.232 |
| Recursive + pruning | 20.293 ± 0.323 |
| Recursive + pruning + hindsight | 17.481 ± 0.503 |

All methods reach 100% final greedy success with 16-step paths. Composition alone
does not improve on flat options here; pruning gives modest changes. In the paired
hindsight experiment, late-training episode length drops by 13.86%, with an
improvement in all five seeds. Hindsight adds about 1,325 retrospective updates per
run on average. Longest observed option activations rise from 2–3 steps without
hindsight to 7–11 with it. These observations are consistent with earlier adoption
and composition of useful options, but do not isolate the cause.

### Nine Rooms findings

| Method | Late-training steps/episode | Completed episodes, mean | Final greedy steps, mean |
| --- | --- | --- | --- |
| Q-learning | 37.76 ± 0.65 | 5,026.2 | 28.00 |
| Flat two-step options | 38.81 ± 0.75 | 4,443.4 | 28.40 |
| Flat + pruning | 36.70 ± 0.96 | 4,461.8 | 28.00 |
| Recursive | 38.00 ± 0.60 | 4,893.8 | 28.00 |
| Recursive + pruning | 37.29 ± 1.32 | 4,727.8 | 28.40 |
| Recursive + pruning + hindsight | 36.12 ± 1.22 | 4,829.0 | 28.80 |

Every method reaches 100% final greedy success. The full method has the shortest
late-training episodes, while Q-learning completes more episodes and reaches more
goals over the whole budget. Q-learning finds optimal 28-step final paths in every
seed; the full method retains 30-step paths in two seeds. Its longest executed
options are only 2–3 steps. Larger stored depths do not establish longer behaviour.

### Reading the results

Each `results/comparison*/` directory contains `episodes.csv`, `evaluation.csv`,
`summary.csv`, `learning_curves.png`, the exact `config.yaml`, and a written report.
The runner regenerates CSVs/configs/learning curves; reports and extra analysis
figures are saved research artifacts and are not automatically regenerated.
Rerunning a config overwrites its generated files under `results/<config-stem>/`;
use a differently named config to retain the historical outputs.

Late training includes completed episodes ending in the last 20% of the primitive
step budget. The final partial episode is recorded but excluded from those metrics.
Final evaluation runs 50 greedy episodes per seed on a copy with learning,
discovery, pruning, and hindsight disabled. Learning curves show the training
policy in 20 equal step bins, with mean ± one sample SD across seeds; they are not
periodic greedy evaluations.

The comparisons match environment steps, not compute, memory, or update counts.
Hindsight adds replay and storage. Applying epsilon at manager decision points also
changes primitive exploration as option duration changes. Successful-only replay,
fixed starts, deterministic maps, and five seeds limit the conclusions. There is
no equal-update ordinary-replay control or demonstrated generalization to unseen
layouts. Recorded timing is not a reliable method comparison because some jobs
overlapped.

## Inspect learned options

```powershell
python visualize_options.py --config configs/comparison_hindsight.yaml --agent composed_pruned_hindsight --seed 3
python visualize_options.py --config configs/comparison_nine_rooms.yaml --agent composed_pruned_hindsight --seed 0
```

This reruns seeded training and writes to
`results/option_snapshots/[nine_rooms/]<mode>/seed_<N>/`:

Softmax modes add `_softmax_T<temperature>` to the mode name, keeping their
outputs separate from the default-policy snapshots.

- `options.html`: standalone viewer with a training-stage slider and policy details.
- `overview.png`: option initiation locations and endpoint connections over time.
- `option_details.png`: representative primitive paths of six final options.
- `snapshots.json`: policies, counts, manager values, hierarchy, and pruning history.
- `representative_rollouts.json`: the plotted representative paths.

`--agent` explicitly selects the method; otherwise the config's feature switches
apply. `--checkpoints 0 1000 5000` overrides snapshot steps and trains through the
largest checkpoint. Nine Rooms has checkpoints in its comparison config; Four
Rooms defaults to eight stages through 100,000 steps.

Snapshot collection does not consume action RNG or change usage. Overview arrows
connect endpoints and are not executed paths. Representative rollouts break modal
ties by lowest action ID; the live agent samples ties. The Four Rooms seed-3
hindsight example was selected after inspecting runs for its long option, whereas
Nine Rooms uses seed 0 consistently across methods. See the reports for all-seed
details and independent snapshot/benchmark agreement.

## Validation

```powershell
python -m unittest discover -v
```

Tests cover room connectivity and shortest paths, collisions and time limits,
seeded starts, nested option execution, discounted targets, pruning/remapping,
hindsight matching and side effects, softmax probabilities and sampling,
random-agent training, and comparison budget
cutoffs. During cleanup, all six main variants were also checked against the
original implementation on a seeded 3,000-step task: every training/evaluation row
and every non-timing summary value matched exactly. The historical full-budget
experiments were retained rather than rerun as part of that check.

## DQN and neural outcome options

`deep_agents.py` supplies a NumPy DQN (two ReLU hidden layers, replay, target network,
Huber loss and Adam), and `OutcomeDQNAgent`. The latter discovers the most common
unseen physical displacement in two-decision segments of successful trajectories,
then clones the demonstrations with a neural policy. Options can call older options,
initiate in every state, and stop relative to their invocation position or at their
bounded duration. Policy-compatible HAR inserts semi-Markov option transitions into
the DQN replay buffer. Goal position is observed for random-goal tasks.

Run the matched nine-room comparisons and generate the report:

```powershell
python compare_agents.py --config configs/dqn_nine_fixed.yaml
python compare_agents.py --config configs/dqn_nine_random.yaml
python analyze_deep_agents.py
```

Each compares standard DQN, outcome DQN + deep BC + HAR, Q-learning, and the earlier
composed + pruned + HAR agent across three seeds, 100,000 environment steps each,
and 100 frozen greedy evaluations. Results and curves are in
`results/dqn_nine_fixed/` and `results/dqn_nine_random/`; the combined report is
`results/dqn_nine_report.md`. The NumPy implementation requires no added packages.
