# Nine Rooms method comparison

## Environment and protocol

NineRoomsEnv inherits the Four Rooms dynamics and generalizes its five-cell room layout to a 3x3 arrangement. The 17x17 grid has 237 walkable states, twelve centered doorways, a fixed start at (1,1), and a fixed goal at (15,15). Actions are up/right/down/left; invalid moves leave the agent in place. Reward is 1 only on reaching the goal; otherwise 0. The episode limit is 500 primitive steps. BFS confirms that the shortest start-to-goal path is 28 steps. Existing Four Rooms geometry is unchanged.

Six methods, five paired seeds (0-4), 300,000 primitive environment steps per run. Alpha=0.1, gamma=0.99, epsilon=0.2. Option methods use 20 active slots, discovery every 10 episodes, minimum 3 occurrences, and a 100-successful-episode buffer. Pruning threshold: 10,000 inactive primitive steps. Hindsight: up to 100 new-option manager updates from compatible successful primitive replay. No per-method tuning or post-hoc budget extension.

Final evaluation uses a copy, 50 greedy episodes per seed, and disables learning, discovery, pruning, and hindsight. Same environment-step budgets do not imply identical learning-update or compute budgets; hindsight uses extra replay and storage. Wall-clock times are not compared because snapshot jobs overlapped some runs.

## Results

Values are mean +/- sample SD across seeds, except length ranges:

| Agent | Late-training steps/episode | Completed episodes | Goals reached | Final greedy steps | Greedy success | Longest observed option per seed |
| --- | --- | --- | --- | --- | --- | --- |
| Q-learning | 37.76 +/- 0.65 | 5026.2 +/- 980.7 | 4817.0 | 28.00 +/- 0.00 | 100% | [0, 0, 0, 0, 0] |
| Two-step | 38.81 +/- 0.75 | 4443.4 +/- 436.9 | 4197.2 | 28.40 +/- 0.89 | 100% | [2, 2, 2, 2, 2] |
| Two-step + pruning | 36.70 +/- 0.96 | 4461.8 +/- 887.3 | 4199.8 | 28.00 +/- 0.00 | 100% | [2, 2, 2, 2, 2] |
| Recursive | 38.00 +/- 0.60 | 4893.8 +/- 323.4 | 4674.4 | 28.00 +/- 0.00 | 100% | [2, 2, 2, 2, 2] |
| Recursive + pruning | 37.29 +/- 1.32 | 4727.8 +/- 412.3 | 4492.0 | 28.40 +/- 0.89 | 100% | [3, 2, 3, 2, 2] |
| Recursive + pruning + hindsight | 36.12 +/- 1.22 | 4829.0 +/- 384.1 | 4592.0 | 28.80 +/- 1.10 | 100% | [3, 3, 3, 2, 3] |

Late training is the last 20% of the step budget, using completed episodes; partial final episodes are excluded. learning_curves.png shows training success and episode length in 15,000-step bins; shaded bands are one sample SD. learning_summary.png adds cumulative task reward (number of goals reached) and a late-training zoom.

The full method has the shortest average late-training episodes, but Q-learning reaches more goals and completes more episodes over the total budget. All methods achieve 100% final success. Q-learning is optimal (28 steps) in all five seeds. The original two-step agent and recursive agent with pruning each produce a 30-step greedy path in seed 0; the full method does so in seeds 0 and 1. The other final greedy paths are 28 steps.

Unlike the Four Rooms result, hindsight does not lead to long executed options here: the full method reaches only 2-3 primitive steps. Nominal hierarchy depth can still be larger, because stored child references do not imply that those calls dominate the cloned policy. Possible constraints include the unchanged 20-option capacity, frequency-based segment selection, and state-specific initiation on a larger grid; these mechanisms have not been isolated by this comparison. Five seeds and one fixed-start layout are insufficient for general superiority claims.

## Learned options and validation

- repetition, seed 0: option 4, initiation [3, 8], target [2, 7], observed lengths/counts {'2': 312}.
- repetition_pruned, seed 0: option 4, initiation [3, 8], target [2, 7], observed lengths/counts {'2': 322}.
- composed, seed 0: option 4, initiation [3, 8], target [2, 7], observed lengths/counts {'2': 159}.
- composed_pruned, seed 0: option 32, initiation [7, 13], target [9, 14], observed lengths/counts {'3': 197}.
- composed_pruned_hindsight, seed 0: option 32, initiation [7, 13], target [9, 14], observed lengths/counts {'3': 169}.
- composed_pruned_hindsight, seed 1: option 29, initiation [13, 14], target [15, 15], observed lengths/counts {'3': 214}.
- composed_pruned_hindsight, seed 2: option 31, initiation [1, 1], target [0, 3], observed lengths/counts {'3': 1231}.
- composed_pruned_hindsight, seed 3: option 3, initiation [0, 1], target [1, 0], observed lengths/counts {'2': 351}.
- composed_pruned_hindsight, seed 4: option 41, initiation [7, 2], target [10, 2], observed lengths/counts {'3': 192}.

Snapshot runs are independent seeded repetitions; final episode counts, pruning counts, hindsight update counts and longest observed lengths match the benchmark. Seed 0 is used consistently to illustrate all option methods; the full method also has snapshots for every seed. These figures were not chosen to maximize option length.

overview.png shows starts and targets at eight training stages; dashed arrows are endpoint connections, not primitive paths. option_details.png shows the six longest-used final options as numbered primitive paths, with coordinates. They are representative rollouts of the saved policies with modal ties broken by the lowest action ID; they are not a claim that every stochastic tie produces the same route. options.html provides a stage slider and policy/child-option inspection; snapshots.json and representative_rollouts.json contain raw data.

Validation: 17 unit tests passed, including preservation of Four Rooms, nine rooms of 25 cells each, 12 connecting doorways, full map connectivity, BFS shortest-path execution/reward checks, collisions, time limits, and seeded random starts. Snapshot references and PNG outputs were validated.

## Reproduce

python compare_agents.py --config configs/comparison_nine_rooms.yaml

python visualize_options.py --config configs/comparison_nine_rooms.yaml --agent composed_pruned_hindsight --seed 0

python train.py --config configs/config_nine_rooms.yaml
