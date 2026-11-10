# Inactivity pruning experiment

Five paired seeds (0-4), 100,000 primitive environment steps each, fixed start (1,1), fixed goal (9,9), 200-step episode limit. Alpha=0.1, gamma=0.99, epsilon=0.2. Discovery every 10 episodes, minimum 3 segment occurrences, at most 20 active options, replay buffer size 100. Pruning threshold chosen before running: 10,000 primitive steps without use. No tuning.

Pruning is opt-in. It runs at episode-end discovery intervals. Newly created options receive a full inactivity grace period. Both manager selections and nested calls count as use; running options refresh their last-used time. Dependencies of recently used options are protected. Pruning compacts Q columns and remaps child calls and replay consistently; replay is split at removed calls to preserve temporal continuity. Stable option IDs are never reused. Surviving manager values are preserved; replacement options start at zero. Previously pruned behaviours can be rediscovered from retained successful experience.

Final evaluation freezes learning, discovery, and pruning on a copy and runs 50 greedy episodes per seed. Timing is not compared because separate snapshot runs overlapped with some training runs.

| Agent | Late-training steps/episode (mean +/- SD) | Completed training episodes (mean) | Greedy steps | Greedy success | Options pruned (mean) |
| --- | --- | --- | --- | --- | --- |
| Q-learning | 21.758 +/- 0.494 | 3646.6 | 16.0 | 100% | 0.0 |
| Two-step | 20.418 +/- 0.403 | 4097.0 | 16.0 | 100% | 0.0 |
| Two-step + pruning | 20.264 +/- 0.421 | 4120.0 | 16.0 | 100% | 8.4 |
| Recursive | 20.544 +/- 0.232 | 4033.0 | 16.0 | 100% | 0.0 |
| Recursive + pruning | 20.293 +/- 0.323 | 4068.8 | 16.0 | 100% | 6.6 |

Late training uses completed episodes ending in the last 20% of the step budget; partial final episodes are excluded. Learning curves show training-policy metrics in 5,000-step bins, mean +/- one sample SD across seeds.

Pruning modestly reduced average late-training episode length for both option methods. Final greedy performance remained identical. Five seeds on this single fixed-start task do not establish a general performance advantage. Different durations alter the exploration induced by the same manager epsilon.

Longest observed option activations across seeds:
- Recursive without pruning: [2, 2, 2, 2, 3]
- Recursive with pruning: [2, 3, 2, 3, 3]

The longest observed executions remain three primitive steps. Pruning makes capacity available for replacements but does not alter the frequency-based discovery objective or the cloning rule that can bypass child calls.

## Snapshot validation

- Seed 0: 5 pruned, 20 retained, longest execution 2 primitives. Snapshot episode counts and removals match the independent comparison run; child and policy references are valid.
- Seed 1: 6 pruned, 20 retained, longest execution 3 primitives. Snapshot episode counts and removals match the independent comparison run; child and policy references are valid.
- Seed 2: 5 pruned, 20 retained, longest execution 2 primitives. Snapshot episode counts and removals match the independent comparison run; child and policy references are valid.
- Seed 3: 7 pruned, 20 retained, longest execution 3 primitives. Snapshot episode counts and removals match the independent comparison run; child and policy references are valid.
- Seed 4: 10 pruned, 20 retained, longest execution 3 primitives. Snapshot episode counts and removals match the independent comparison run; child and policy references are valid.

Results: episodes.csv, evaluation.csv, summary.csv, learning_curves.png, and exact config.yaml. All five recursive-pruning snapshot viewers, PNG overviews, JSON policies and pruning event logs are in results/option_snapshots/composed_pruned/seed_N/.

Reproduce comparison: python compare_agents.py --config configs/comparison_pruning.yaml
Reproduce snapshots: python visualize_options.py --config configs/comparison_pruning.yaml --seed 3
