# Recursive option comparison

Five paired seeds (0-4); 100,000 primitive environment steps per run; same fixed start and goal, learning rates, epsilon, discovery interval, and 20-option cap. No tuning. Final greedy evaluation uses 50 episodes per seed.

| Metric (mean +/- sample SD) | Q-learning | Two-step options | Recursive options |
| --- | --- | --- | --- |
| Completed training episodes | 3646.600 +/- 356.738 | 4097.000 +/- 277.690 | 4033.000 +/- 159.839 |
| Late training steps/episode | 21.758 +/- 0.494 | 20.418 +/- 0.403 | 20.544 +/- 0.232 |
| Greedy success fraction | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Greedy steps/episode | 16.000 +/- 0.000 | 16.000 +/- 0.000 | 16.000 +/- 0.000 |

Late training is the last 20% of environment steps, using completed episodes. Curves show 5,000-step bins, mean +/- one SD across seeds. These are training-policy curves (epsilon=0.2), not periodic greedy evaluations.

Recursive composition did not outperform the two-step agent on average here. Both use fewer late-training steps than Q-learning, but all tie in final greedy performance. The same epsilon at manager decisions changes primitive-level exploration with option duration, so this does not isolate abstraction from exploration.

Recursive depth by seed: [1, 2, 2, 3, 2]. The 20-option cap is filled early, which prevents later growth. Policies only refer to older options; their internal decisions remain state dependent. A safe primitive-step limit is not the same as an observed execution length.

## Saved snapshot runs

Seed 0: 0 of 20 options call other options.

Seed 3: 2 of 20 options call other options.
- Option 5: children [2], depth 2, primitive-step safety limit 4, observed execution lengths (steps: count) {'2': 29}.
- Option 6: children [3, 5], depth 3, primitive-step safety limit 8, observed execution lengths (steps: count) {'1': 15}.

The depth-3 seed was selected for illustration after examining all seeds; seed 0 is retained as an example with no composite discoveries. This is not evidence that depth 3 is typical.

Rerun: python compare_agents.py --config configs/comparison_recursive.yaml
Snapshots: python visualize_options.py --config configs/comparison_recursive.yaml --seed 3

## Actual execution lengths across all five seeds

| Seed | Composite policies | Max depth | Longest observed primitive execution |
| --- | --- | --- | --- |
| 0 | 0 | 1 | 2 |
| 1 | 3 | 2 | 2 |
| 2 | 2 | 2 | 2 |
| 3 | 2 | 3 | 2 |
| 4 | 1 | 2 | 3 |

Execution lengths count top-level option activations, including early terminations. Depth describes nested policy references, not a fixed number of primitive actions. All five seeded snapshot runs are saved under results/option_snapshots/composed/.
