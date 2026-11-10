# Repetition agent vs Q-learning

Five paired seeds (0?4), 100,000 primitive environment steps per run. Fixed start (1,1), goal (9,9), 200-step episode limit. Alpha=0.1, gamma=0.99, epsilon=0.2. No hyperparameter tuning.

Values below are mean ? sample SD across seeds. Final training metrics use completed episodes ending in the last 20% of the step budget. Evaluation uses 50 greedy episodes per seed on copies of the learned agents; learning and discovery are disabled.

| Metric | Q-learning | Repetition |
| --- | --- | --- |
| Completed training episodes | 3646.600 ? 356.738 | 4097.000 ? 277.690 |
| Training success fraction | 0.971 ? 0.017 | 0.981 ? 0.007 |
| Final training steps per episode | 21.758 ? 0.494 | 20.418 ? 0.403 |
| Greedy success fraction | 1.000 ? 0.000 | 1.000 ? 0.000 |
| Greedy steps per episode | 16.000 ? 0.000 | 16.000 ? 0.000 |
| Discovered options | 0.000 ? 0.000 | 20.000 ? 0.000 |

Both agents converge to the same final greedy performance. Repetition achieves shorter training episodes on average in this setting. This is an initial five-seed result on one fixed-start task, not evidence of general superiority. Exploration selects options at manager decision points, so matching epsilon does not produce identical primitive-action exploration. The implementation uses one level of two-step options; it does not yet test recursive hierarchy construction.

learning_curves.png shows per-seed averages in 5,000-step bins, then mean ? one sample standard deviation across seeds. Partial final episodes are excluded from episode metrics. Raw training and evaluation rows and the exact config are saved alongside this report.

Rerun from the project root with: python compare_agents.py
