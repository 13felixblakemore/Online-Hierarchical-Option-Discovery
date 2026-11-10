# Recursive macro-actions with Hindsight Action Replay

## Paper guidance and implementation

Guidance: Christodoulou, Lange, Shafti and Faisal (2019), [Reinforcement Learning with Structured Hierarchical Grammar Representations of Actions](https://arxiv.org/abs/1910.02876). The paper introduces AG-RL and Hindsight Action Replay (HAR). This repo implements a tabular adaptation inspired by grammar-based action expansion and HAR, not a reproduction of its Atari agents.

Our implementation stores binary rules over primitives or older macro symbols. It compresses each successful primitive trajectory with existing rules, counts repeated digrams, and adds one qualifying nonoverlapping pair every 10 episodes. Occurrence threshold is 3 and a new rule must save tokens after a two-token RHS cost. This is a greedy repeated-pair compressor, not exact Sequitur or the paper’s information-theoretic grammar criterion. Episodes are never concatenated for matching. Repeated primitive expansions are deduplicated. Rules reference only older rules, making recursion acyclic. A cached expansion supports matching, while runtime execution traverses the recursive rule stack.

Macros are fixed open-loop action sequences available in every state and under every task goal. They execute to sequence completion, true terminal goal or time limit. They do not have a spatial subgoal, cloned state policy or unknown-state termination. A wall consumes an action and leaves position unchanged; global availability does not guarantee useful navigation.

The tabular manager learns actual discounted duration returns with gamma**duration bootstrapping; only true terminal states remove the bootstrap. Primitive Q-values learn on every step, even inside macros. New macro columns copy the value of their first primitive, preserving existing columns. The HAR version additionally relabels exact observed primitive sequences as macro executions at their actual start observation and replays a new macro against up to 100 compatible segments from the latest 100 completed episodes. Goal-interrupted matching prefixes are valid; nonterminal incomplete prefixes are not relabelled. At most one longest compatible suffix per macro per step is used online. The actual selected-macro transition is not duplicated. Rewards and goals stay unchanged, and episode boundaries are never crossed.

Differences from the paper: training successful trajectories replace separate elite greedy grammar rollouts; tabular online updates replace neural replay training; no action-balanced replay, Abandon Ship heuristic or macro exploration bonus. These choices keep the current exploration settings and environment budgets. Macro ablation disables macro relabelling/warm-start replay but retains primitive learning inside macros, value copying and the same induction procedure. Its learned grammar can differ because training trajectories differ. This is not an equal-update control.

## Matched experiments

Five seeds (0–4). Four Rooms fixed task: 100,000 steps and 50 evaluations/seed. Nine Rooms fixed task: 300,000 steps and 50 evaluations/seed. Four Rooms random distinct start/goal with both positions observed: 1,000,000 steps and 1,000 matched evaluations/seed. Alpha=.1, gamma=.99, epsilon=.2, buffers=100, discovery interval=10, minimum occurrences=3. Library cap=20, macro maximum length=32. Rules are appended without pruning and freeze once capacity is full. The cloned-option baseline retains its original pruning. Settings were fixed before running; no tuning or post-result budget extension.

Existing Q-learning and cloned-option baselines are reused after exact config/variant/seed/budget validation. Raw data are included in each new comparison directory. Baseline file hashes and package versions are in runtime.json. Macro and ablation runs are new. Some jobs ran concurrently, so wall-clock times are not compared. Equal environment steps do not imply equal computation, memory or learning updates. HAR adds extra updates; availability and duration change exploration.

Final evaluation freezes all learning and grammar induction on copied agents. Late training uses complete episodes ending in the final 20% of each budget; partial final episodes are excluded. Curves use 20 equal step bins, mean ± sample SD across trained seeds, and are training-policy curves. Random-task training prefixes and evaluation panels are checked for pairing; evaluation panels overlap across seeds under the existing seed formula. BFS distances are for analysis only.

## Results

| Task | Agent | Late-training steps | Training goals | Final success fraction | Final steps | Final SPL |
| --- | --- | --- | --- | --- | --- | --- |
| Four Rooms, fixed task | Q-learning | 21.758 ± 0.494 | 3547.400 ± 400.524 | 1.000 ± 0.000 | 16.000 ± 0.000 | 1.000 ± 0.000 |
| Four Rooms, fixed task | Cloned options + hindsight | 17.481 ± 0.503 | 4589.800 ± 365.733 | 1.000 ± 0.000 | 16.000 ± 0.000 | 1.000 ± 0.000 |
| Four Rooms, fixed task | Macros, no macro replay | 27.041 ± 4.193 | 3245.800 ± 550.538 | 1.000 ± 0.000 | 16.000 ± 0.000 | 1.000 ± 0.000 |
| Four Rooms, fixed task | Macros + hindsight action replay | 19.679 ± 3.086 | 4739.400 ± 708.801 | 1.000 ± 0.000 | 16.000 ± 0.000 | 1.000 ± 0.000 |
| Nine Rooms, fixed task | Q-learning | 37.763 ± 0.646 | 4817.000 ± 1055.470 | 1.000 ± 0.000 | 28.000 ± 0.000 | 1.000 ± 0.000 |
| Nine Rooms, fixed task | Cloned options + hindsight | 36.124 ± 1.215 | 4592.000 ± 405.347 | 1.000 ± 0.000 | 28.800 ± 1.095 | 0.973 ± 0.037 |
| Nine Rooms, fixed task | Macros, no macro replay | 40.836 ± 2.866 | 6008.600 ± 450.541 | 1.000 ± 0.000 | 28.000 ± 0.000 | 1.000 ± 0.000 |
| Nine Rooms, fixed task | Macros + hindsight action replay | 34.501 ± 3.501 | 7166.000 ± 1017.444 | 1.000 ± 0.000 | 28.800 ± 1.095 | 0.973 ± 0.037 |
| Four Rooms, random start/goal | Q-learning | 12.551 ± 0.084 | 29493.600 ± 1330.162 | 1.000 ± 0.000 | 9.074 ± 0.046 | 0.914 ± 0.006 |
| Four Rooms, random start/goal | Cloned options + hindsight | 12.600 ± 0.063 | 28839.000 ± 651.522 | 1.000 ± 0.000 | 9.067 ± 0.053 | 0.916 ± 0.005 |
| Four Rooms, random start/goal | Macros, no macro replay | 38.881 ± 29.418 | 18972.400 ± 12495.704 | 0.910 ± 0.131 | 29.025 ± 26.204 | 0.715 ± 0.134 |
| Four Rooms, random start/goal | Macros + hindsight action replay | 29.364 ± 26.686 | 23929.000 ± 13030.204 | 0.934 ± 0.139 | 23.395 ± 26.780 | 0.758 ± 0.127 |

SPL is success * shortest_distance / max(shortest_distance, executed_steps). Failures score zero. Final step counts include failures at their time limit. Sample SD describes five-seed variability; no significance claim is made.

### Measured comparisons

- Four Rooms, fixed task: HAR late-training steps change -9.56% versus Q-learning, +12.57% versus cloned options, and -27.23% versus macros without macro replay. HAR reaches 4739.4 training goals on average versus 3547.4 for Q-learning. Final HAR success is 100.00% and mean path length 16.000. Its final SPL changes +0.0000 versus Q-learning.
- Nine Rooms, fixed task: HAR late-training steps change -8.64% versus Q-learning, -4.49% versus cloned options, and -15.51% versus macros without macro replay. HAR reaches 7166.0 training goals on average versus 4817.0 for Q-learning. Final HAR success is 100.00% and mean path length 28.800. Its final SPL changes -0.0267 versus Q-learning.
- Four Rooms, random start/goal: HAR late-training steps change +133.96% versus Q-learning, +133.04% versus cloned options, and -24.48% versus macros without macro replay. HAR reaches 23929.0 training goals on average versus 29493.6 for Q-learning. Final HAR success is 93.42% and mean path length 23.395. Its final SPL changes -0.1564 versus Q-learning.

Training-policy efficiency and frozen final path efficiency are different outcomes. A long macro can reduce manager decisions and still take detours. HAR improves data reuse but adds many more updates; no equal-update ordinary-replay control is included.

Random-task final HAR success by seed is [100.0, 98.6, 100.0, 68.5, 100.0] percent. This substantial seed variability is hidden by a single average. Fixed-goal final HAR paths are 16 steps in every Four Rooms seed; Nine Rooms has 30-step paths in seeds 2 and 4, while Q-learning has optimal 28-step paths in every seed. HAR improves average late-training episode length over the macro ablation in all three tasks, but does not remove the random-task failures or establish a general performance advantage.

## Actual macro use and replay

| Task | Agent | Longest observed execution per seed | Stored maximum depth per seed | Online macro HAR updates | Discovery HAR updates |
| --- | --- | --- | --- | --- | --- |
| Four Rooms, fixed task | Macros, no macro replay | [14, 20, 16, 22, 10] | [5, 6, 5, 6, 4] | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Four Rooms, fixed task | Macros + hindsight action replay | [18, 17, 18, 22, 16] | [5, 6, 5, 6, 6] | 135409.400 ± 31278.981 | 1696.800 ± 217.308 |
| Nine Rooms, fixed task | Macros, no macro replay | [9, 6, 16, 11, 14] | [4, 3, 5, 5, 5] | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Nine Rooms, fixed task | Macros + hindsight action replay | [30, 11, 26, 12, 8] | [8, 5, 6, 5, 3] | 283238.400 ± 49817.393 | 1880.200 ± 82.482 |
| Four Rooms, random start/goal | Macros, no macro replay | [8, 5, 7, 5, 8] | [3, 3, 4, 3, 3] | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Four Rooms, random start/goal | Macros + hindsight action replay | [5, 5, 8, 6, 8] | [3, 3, 3, 3, 3] | 830164.800 ± 138163.306 | 1970.200 ± 25.104 |

Unlike the earlier cloned policies, macro definitions do not contain absolute position or task-goal keys. The manager still uses the full observed state, including the goal when present, to choose a useful action. This shares the action sequence, not the manager values or a goal-reaching policy across states.

| Task | Physical starting cells where some HAR macro was selected, per seed | Task goals where some HAR macro was selected, per seed |
| --- | --- | --- |
| Four Rooms, fixed task | [103, 94, 93, 100, 91] | [1, 1, 1, 1, 1] |
| Nine Rooms, fixed task | [236, 236, 226, 233, 235] | [1, 1, 1, 1, 1] |
| Four Rooms, random start/goal | [104, 104, 104, 104, 104] | [104, 104, 104, 104, 104] |

## Qualitative examples

Seed 0 is fixed for illustration. For each environment, show the most-selected final HAR macro and the one with the longest observed execution (ties by selections). Draw the same stored primitive expansion from up to three most-used observed starts. These are deterministic saved-sequence rollouts in the grid, not additional training. Path overlap can hide repeated wall collisions; legends report blocked moves. Full grammars, expansions, depths and observed lengths for every seed are in each comparison’s macro*_grammar.md and macros/*.json.

| Task | Selection rule | Rule | Expanded primitive actions | Depth | Top-level selections |
| --- | --- | --- | --- | --- | --- |
| four_rooms | most_selected | M14 → M11 M13 | →↓↓↓↓↓↓←↓→→→→→↓→→→ | 5 | 2989 |
| four_rooms | longest_executed | M14 → M11 M13 | →↓↓↓↓↓↓←↓→→→→→↓→→→ | 5 | 2989 |
| nine_rooms | most_selected | M19 → M16 M18 | ↓↓→→→↑→→↓↓→→↓↓→→↓↓→→↓↓→→↓↓→↓↓↓ | 8 | 3534 |
| nine_rooms | longest_executed | M19 → M16 M18 | ↓↓→→→↑→→↓↓→→↓↓→→↓↓→→↓↓→→↓↓→↓↓↓ | 8 | 3534 |
| random_goals | most_selected | M2 → ↓ ← | ↓← | 1 | 31111 |
| random_goals | longest_executed | M19 → M4 M1 | ↑↑←→→ | 3 | 16520 |

## Limits and reproduction

A sequence can be executable everywhere and still hit walls or take unnecessary detours. The current implementation has no feedback-based abort except episode termination. A library filled early can preserve poor rules; we did not add pruning or tune capacity afterward. HAR can correct manager values but cannot change a stored sequence. Five seeds and these layouts do not establish generalization to unseen maps or replicate the paper’s Atari results.

```powershell
python compare_agents.py --config configs/comparison_macros_four_rooms.yaml
python compare_agents.py --config configs/comparison_macros_nine_rooms.yaml
python compare_agents.py --config configs/comparison_macros_random_goals.yaml
python analyze_macro_actions.py
```

Remove experiment.reuse_baselines_from to retrain baselines. All original result folders are preserved. Validation: 44 tests passed; recursive expansions, acyclic grammar references, length limits, snapshot counters, exact budgets, and paired random tasks are checked. Frozen evaluation does not learn or induce rules. Every benchmark macro library is saved before evaluation.
