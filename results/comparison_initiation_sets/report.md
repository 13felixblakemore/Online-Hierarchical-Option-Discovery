# Broader initiation-set experiment

## What changed

The original singleton initiation set is a valid option initiation set, but restrictive. The new policy_support mode uses every demonstrated policy state except the option target. It does not invent actions at unseen states. Both manager availability and bootstrap maxima use the same set; hindsight matches from every allowed start and updates that actual start row. Termination limits, frequency-based discovery, cloning, pruning and manager learning are unchanged.

This isolates expanding the initiation set over existing policy coverage. It does not introduce a new policy architecture, goal-independent skills, a model-based planner or a fallback navigator. Membership means a defined initial policy action, not a guarantee of reaching the target. Cloned policies can still terminate early or exhaust their budget; the probes below measure this.

## Protocol

Four Rooms, random distinct start and task goal each episode; observations include both positions as position_cell*104+goal_cell. Five paired seeds 0–4, 1,000,000 primitive steps each, 200-step episode limit. Alpha=.1, gamma=.99, epsilon=.2. Most-frequent option policy. 20 global active slots, discovery every 10 episodes, minimum 3 occurrences, buffer size 100, pruning after 10,000 inactive steps, up to 100 hindsight updates per new option. These settings were fixed before running, with no tuning or budget extension.

Q-learning and single-state baselines are reused from comparison_random_goals, not freshly trained here. The runner validates exact environment, variant settings, seed list and budgets before reuse. Remove reuse_baselines_from from the config to rerun all three. Raw baseline results are copied into this experiment; historical files remain unchanged. Timing is not compared.

Final evaluation: 1,000 matching tasks per seed, copied agents with learning, discovery, pruning and hindsight disabled. All training task sequence prefixes and all evaluation panels were checked for agreement. Evaluation panels overlap across seeds under the existing seed formula; SD is across five trained agents. Curves are training-policy metrics in 50,000-step bins, not periodic evaluations.

## Results

| Method | Evaluation success | Evaluation steps | Evaluation SPL | Late-training steps | Training goals |
| --- | --- | --- | --- | --- | --- |
| Q-learning | 1.000 ± 0.000 | 9.074 ± 0.046 | 0.914 ± 0.006 | 12.551 ± 0.084 | 29493.600 ± 1330.162 |
| Recursive + pruning + hindsight | 1.000 ± 0.000 | 9.067 ± 0.053 | 0.916 ± 0.005 | 12.600 ± 0.063 | 28839.000 ± 651.522 |
| Recursive + pruning + hindsight + initiation set | 1.000 ± 0.000 | 9.061 ± 0.047 | 0.920 ± 0.004 | 12.613 ± 0.139 | 28380.200 ± 1769.298 |

SPL = success * shortest_path / max(shortest_path, actual_steps). BFS is used for reporting only and is not given to the agents. Late training uses completed episodes ending in the last 20% of the step budget; partial final episodes are excluded.

| Option method | Options created | Options pruned | Hindsight updates | Longest execution per seed | Final depth per seed |
| --- | --- | --- | --- | --- | --- |
| Recursive + pruning + hindsight | 1256.600 ± 12.300 | 1236.600 ± 12.300 | 5776.400 ± 144.266 | [2, 2, 2, 2, 2] | [1, 1, 1, 1, 1] |
| Recursive + pruning + hindsight + initiation set | 1209.600 ± 15.758 | 1189.600 ± 15.758 | 8248.800 ± 245.483 | [3, 2, 2, 2, 2] | [1, 1, 1, 1, 1] |

Paired evaluation SPL changes, broad minus singleton: [0.00264, 0.01091, -0.00105, 0.00502, -0.00112].

## Initiation and all-states probes

For every final broad-set option, probe every physical cell under its original task-goal context. Exclude the task terminal and option target cells. Each remaining start has 32 seeded policy rollouts. Availability is bypassed for the all-states diagnostic; an undefined policy stops immediately and counts as failure. There is no fallback action, shortest-path controller, learning or extra training interaction. Dynamics are used only to execute the stored policy.

Results below average the per-option rates within each seed. These diagnostics concern reaching an option subgoal, not whole-task success. Finite samples cannot certify success probabilities.

| Seed | Mean initiation-set size | Target success from supported starts | From additional supported starts | Forced all-cells success | Undefined at start |
| --- | --- | --- | --- | --- | --- |
| 0 | 2.05 | 100.0% | 100.0% | 2.0% | 98.0% |
| 1 | 2.10 | 100.0% | 100.0% | 2.1% | 97.9% |
| 2 | 2.20 | 100.0% | 100.0% | 2.1% | 97.9% |
| 3 | 2.05 | 100.0% | 100.0% | 2.0% | 98.0% |
| 4 | 2.05 | 100.0% | 100.0% | 2.0% | 98.0% |

## What transfers, and what does not

A stored policy can be reused from an additional state when it already defines an action there; starting mid-demonstration can execute a useful suffix. This is reuse of demonstrated coverage, not generalization to unobserved positions. Merely exposing an option everywhere does not create the missing policy entries. An option also remains specific to its original task goal because goal identity is part of every policy key. It is unavailable under a different goal even at the same physical cell.

To support all nonterminal positions meaningfully, a separate extension would need a policy defined across the map: for example a learned option-subgoal-conditioned controller trained from diverse starts. Its initiation set should then reflect measured reachability or success under its time budget. Relative features can share behaviour across positions, but walls and doorways mean a translated action sequence need not remain valid. That architecture change is deliberately excluded from this initiation-only ablation.

The 20-option global capacity and goal-specific buffers remain unchanged. Hindsight now has more eligible start states, so its update count can change; equal environment steps do not mean equal compute or updates. Options also alter primitive exploration under manager epsilon. Five seeds, one layout and in-distribution tasks do not establish general superiority.

## Reproduce

```powershell
python compare_agents.py --config configs/comparison_initiation_sets.yaml
python analyze_initiation_sets.py
```

Artifacts: learning_curves.png (all methods), learning_summary.png (adds goals and SPL), initiation_sets.png (seed-0 examples), raw CSVs, aggregate.json, saved option policies in options/, and transfer_probes.json. Seed 0 and the six most-selected final options were specified before inspecting results. Validation: 32 tests passed before training; 33 tests including an additional saved-policy probe check passed after training. Manager availability, replay from additional starts, goal context and baseline setting matching are checked. The analysis checks matched tasks and probes every final option.
