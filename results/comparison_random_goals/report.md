# Random-start, random-goal Four Rooms comparison

## Setup

Q-learning versus composed + pruned + hindsight, most-frequent option policy. Five paired seeds (0–4), 1,000,000 primitive environment steps per method/seed, fixed before running. The budget is larger than the earlier fixed-goal experiments because the joint state space has 10,816 states. No tuning or budget extension after viewing results.

Both endpoints are sampled uniformly among distinct walkable cells on every reset: goal uniformly among 104 cells, start uniformly among the remaining 103. Goals stay fixed within an episode. Observations encode position_cell * 104 + goal_cell; divmod(observation,104) recovers both cell IDs, which map to (row,column) through the environment positions list. Both agents receive the same joint observation and tabular representation. No shortest-path information is given to either agent.

Reward is 1 on reaching the sampled goal and otherwise 0. Episode limit 200; truncations bootstrap. Alpha=0.1, gamma=0.99, epsilon=0.2. Option settings match the earlier full method: 20 active slots globally, discovery every 10 episodes, minimum 3 occurrences, 100-successful-episode buffers, 10,000-step inactivity pruning, at most 100 hindsight updates per new option.

Both methods see the same seeded training task sequence by episode number; different episode lengths mean different numbers of tasks within the step budget. This prefix agreement was checked for all seeds. Final evaluation uses 1,000 exactly matched random tasks per seed with manager exploration disabled and no learning, discovery, pruning or hindsight on copied agents. Modal option ties and Q-value ties retain random tie breaking. Evaluation seed formula is 10000 + 100*training_seed + episode_index; panels overlap across training seeds and repeated tasks are possible. Seed SD is across trained agents, not 5,000 independent evaluation panels.

## Results

| Metric (mean ± sample SD across five seeds) | Q-learning | Composed + pruned + hindsight |
| --- | --- | --- |
| Final evaluation success | 100.000 ± 0.000% | 100.000 ± 0.000% |
| Evaluation steps including failures (200-step cap) | 9.074 ± 0.046 | 9.067 ± 0.053 |
| Evaluation SPL | 0.914 ± 0.006 | 0.916 ± 0.005 |
| Fraction of evaluation tasks solved optimally | 66.400 ± 2.032% | 66.740 ± 2.039% |
| Path length / shortest path, successful evaluations only | 1.135 ± 0.011 | 1.131 ± 0.010 |
| Late-training success | 99.999 ± 0.003% | 99.999 ± 0.003% |
| Late-training steps per episode | 12.551 ± 0.084 | 12.600 ± 0.063 |
| Total training goals reached | 29493.600 ± 1330.162 | 28839.000 ± 651.522 |
| Completed training episodes | 31742.200 ± 1245.313 | 31126.200 ± 600.409 |
| Options created | 0.000 ± 0.000 | 1256.600 ± 12.300 |
| Options pruned | 0.000 ± 0.000 | 1236.600 ± 12.300 |
| Extra manager hindsight updates | 0.000 ± 0.000 | 5776.400 ± 144.266 |

Paired final success changes (option agent minus Q-learning), percentage points, seeds 0–4: [0.0, 0.0, 0.0, 0.0, 0.0].

The option agent changes mean final success by +0.000 percentage points, evaluation SPL by +0.0020, and total training goals reached by -654.6 relative to Q-learning. These describe the measured runs, without a significance claim.

Late training uses completed episodes ending in the last 20% of the budget. Partial final episodes are excluded. SPL is success * shortest_path / max(shortest_path, actual_steps); failures score zero. Shortest paths are computed by BFS for reporting only. Success-only path ratios can select different task subsets, so SPL and paired success are the primary evaluation measures. Evaluation is in-distribution across all random tasks, not a held-out-goal or held-out-layout generalization test.

## Option behaviour and interpretation

This experiment uses the existing option algorithm on the joint state space without a new cross-goal abstraction. Option initiation, target, cloned policy and hindsight matching include the episode goal. Replaying a trajectory for a different task goal cannot match an option initiation. Rewards keep their original meaning; no task goals or rewards are relabelled.

This makes each discovered option task-goal-specific. The 20 active slots are shared across 104 possible goals; they are not 20 per goal. Sparse repeated joint-state pairs, successful-only buffers and the global capacity may constrain discovery and reuse, but this comparison does not isolate those mechanisms. Plain tabular Q-learning also has no generalization between goal columns. This is a direct comparison of the existing algorithms with sufficient goal information, not a goal-independent skill-transfer method.

Longest observed top-level option executions, per seed: [2, 2, 2, 2, 2]. Maximum final stored hierarchy depths: [1, 1, 1, 1, 1]. Stored depth is not executed duration.

The agent performs 4.60 hindsight updates per created option on average (ratio of aggregate counts), compared with the configured maximum of 100. This measures the amount of compatible retained replay available at creation, not the potential benefit of replay. Options that do not recur for 10,000 primitive steps are eligible for pruning. Task-goal-specific starts and the global successful-episode buffers make cross-goal reuse unavailable in this implementation.

Primitive environment budgets and learning parameters match. Hindsight adds replay updates and memory; compute is not matched. Manager-level epsilon yields different primitive exploration when options run. Five training seeds and a single deterministic layout limit the conclusions. No significance claim is made.

## Artifacts and reproduction

- `learning_curves.png`: training-policy success and episode lengths.
- `learning_summary.png`: training curves plus cumulative goals and path efficiency.
- `evaluation_by_distance.png`: frozen evaluation performance grouped by task shortest distance.
- `episodes.csv` / `evaluation.csv`: raw task IDs and outcomes.
- `evaluation_paths.csv`: evaluation with BFS shortest distances, optimality and SPL.
- `summary.csv` / `aggregate.json`: per-seed and aggregated metrics.
- `task_validation.json`: task coverage and sampling checks; `runtime.json`: package versions.

```powershell
python compare_agents.py --config configs/comparison_random_goals.yaml
python analyze_random_goals.py
```

Validation: 28 tests passed, including joint observations, varying/reproducible task endpoints, goal consistency within episodes, actual shortest-path goal execution/reward, task pairing and hindsight rejection across task goals. The analysis also checks all training task prefixes, all paired evaluation tasks and exact step budgets.
