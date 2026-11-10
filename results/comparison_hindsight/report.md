# Manager hindsight comparison

## Setup

Five paired seeds (0-4), 100,000 primitive environment steps per run. Both agents use recursive composition and inactivity pruning (10,000-step threshold), 20 active options, discovery every 10 episodes, minimum 3 occurrences, and a 100-successful-episode discovery buffer. Fixed Four Rooms start (1,1), goal (9,9), episode limit 200; alpha=0.1, gamma=0.99, epsilon=0.2. No tuning. The variants override manager_hindsight on/off; the exact experiment configuration is config.yaml.

## Implementation

With manager_hindsight=true, a newly created option receives up to 100 semi-Markov Q-learning updates from a separate buffer of the most recent 100 successful primitive trajectories. Each replay segment begins at the initiation state and must follow modal (maximum-count) policy actions, including any tied choices and recursively called options, through actual option termination. Matching honors parent interruption and early termination. Returns are recomputed from original primitive rewards with gamma discounts. The bootstrap uses gamma to the actual primitive duration; true terminal states bootstrap to zero. No goals or rewards are relabelled.

Updates apply only to the new option column, once at discovery, without changing real selection counts, pruning timers, environment step counts, or action RNG. Pruning never remaps the primitive replay buffer because it contains no option IDs. Evaluation disables learning, discovery, pruning, and hindsight on a copy.

## Results

Mean +/- sample SD across seeds:

| Metric | Hindsight off | Hindsight on |
| --- | --- | --- |
| Late-training steps/episode | 20.293 +/- 0.323 | 17.481 +/- 0.503 |
| Completed training episodes | 4068.800 +/- 171.437 | 4669.600 +/- 348.516 |
| Overall training success fraction | 0.980 +/- 0.006 | 0.983 +/- 0.005 |
| Final greedy success fraction | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Final greedy steps/episode | 16.000 +/- 0.000 | 16.000 +/- 0.000 |
| Extra retrospective manager updates | 0.000 +/- 0.000 | 1324.600 +/- 276.145 |
| Options created | 26.600 +/- 2.074 | 26.600 +/- 3.362 |
| Options pruned | 6.600 +/- 2.074 | 6.600 +/- 3.362 |

Hindsight reduced mean late-training episode length by 13.86%. The paired reductions (off minus on, primitive steps/episode) were [2.489, 3.301, 2.806, 3.517, 1.949]. All five paired runs improved on this metric. Final greedy behaviour remained tied at 100% success and 16 steps.

Late training uses completed episodes ending in the last 20% of the environment-step budget. Partial final episodes are excluded. Evaluation uses 50 greedy episodes per seed. Learning curves plot training-policy success and steps in 5,000-step bins with mean +/- one sample SD; they are not periodic greedy evaluations.

Longest observed option execution per seed:
- Hindsight off: [2, 3, 2, 3, 3]
- Hindsight on: [7, 9, 9, 11, 7]

This is consistent with hindsight allowing useful options to be adopted sooner and subsequently composed. It does not establish that mechanism causally. With epsilon applied at manager decisions, longer options also reduce the number of exploratory decisions per primitive step. Hindsight adds replay updates and storage; compute and memory budgets are not matched. There is no equal-update ordinary-replay control, and successful-only hindsight can bias estimates in stochastic environments. This is an initial result on one deterministic, fixed-start task, not a general performance claim.

## Option snapshots

- Seed 0: option 22, initiation [8, 3], target [9, 9], depth 3, observed lengths/counts {'7': 135}. Counts and hindsight updates match the independent benchmark run.
- Seed 1: option 21, initiation [2, 3], target [6, 8], depth 4, observed lengths/counts {'9': 136}. Counts and hindsight updates match the independent benchmark run.
- Seed 2: option 22, initiation [2, 4], target [7, 8], depth 4, observed lengths/counts {'9': 93}. Counts and hindsight updates match the independent benchmark run.
- Seed 3: option 31, initiation [1, 1], target [8, 5], depth 4, observed lengths/counts {'11': 6}. Counts and hindsight updates match the independent benchmark run.
- Seed 4: option 29, initiation [4, 6], target [8, 9], depth 4, observed lengths/counts {'7': 40}. Counts and hindsight updates match the independent benchmark run.

All five HTML viewers, PNG overviews and raw snapshots are in results/option_snapshots/composed_pruned_hindsight/seed_N/. Snapshot collection does not sample actions or alter usage. Seed 3 is the longest-option example, selected after examining the runs.

## Reproduce

python compare_agents.py --config configs/comparison_hindsight.yaml

python visualize_options.py --config configs/comparison_hindsight.yaml --seed 3

Validation: 13 unit tests passed, covering nested execution, pruning/remapping, discounting, terminal bootstrap, matching rejection, replay limits, opt-in behaviour, and replay side effects. Snapshot episode counts, pruning counts, hindsight updates, and longest observed lengths match independent benchmark runs for all seeds.
