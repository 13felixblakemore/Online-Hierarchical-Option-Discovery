# Four Rooms softmax action selection experiment

## Protocol

Eight methods, five paired seeds (0–4), 100,000 primitive steps per run. Fixed start (1,1), goal (9,9), 200-step limit. Alpha=0.1, gamma=0.99, epsilon=0.2. Discovery every 10 episodes, minimum 3 occurrences, 20 active options, 100 successful episodes in each replay buffer. Pruning threshold 10,000 steps; hindsight limit 100 updates per new option. Softmax uses exp(raw action count / T), T=1 fixed before running, without tuning.

All six original methods were freshly rerun. The only configuration difference in each modal/softmax pair is option action selection. Flat softmax is included to separate the policy change from recursive composition, pruning and hindsight. All shared non-timing summary values of the six baselines exactly match their historical runs. Package versions are in runtime.json.

Final evaluation uses 50 episodes per seed on copies, manager exploration disabled, learning and option changes disabled. Option policies retain their configured stochasticity, including softmax sampling and modal tie breaking. This measures deployed option behaviour, not a determinized softmax policy. Late training means completed episodes ending after step 80,000; partial final episodes are excluded. Curves use 5,000-step bins and mean ± sample SD across seeds.

## Results

| Method | Late training steps | Completed episodes | Greedy success fraction | Greedy steps |
| --- | --- | --- | --- | --- |
| Q-learning | 21.758 ± 0.494 | 3646.600 ± 356.738 | 1.000 ± 0.000 | 16.000 ± 0.000 |
| Two-step options | 20.418 ± 0.403 | 4097.000 ± 277.690 | 1.000 ± 0.000 | 16.000 ± 0.000 |
| Two-step + pruning | 20.264 ± 0.421 | 4120.000 ± 262.051 | 1.000 ± 0.000 | 16.000 ± 0.000 |
| Recursive options | 20.544 ± 0.232 | 4033.000 ± 159.839 | 1.000 ± 0.000 | 16.000 ± 0.000 |
| Recursive + pruning | 20.293 ± 0.323 | 4068.800 ± 171.437 | 1.000 ± 0.000 | 16.000 ± 0.000 |
| Recursive + pruning + hindsight | 17.481 ± 0.503 | 4669.600 ± 348.516 | 1.000 ± 0.000 | 16.000 ± 0.000 |
| Two-step options + softmax | 21.102 ± 1.244 | 3751.400 ± 266.163 | 1.000 ± 0.000 | 16.400 ± 0.894 |
| Recursive + pruning + hindsight + softmax | 18.153 ± 0.951 | 4351.800 ± 276.428 | 1.000 ± 0.000 | 16.400 ± 0.894 |

## Paired policy effects

- Two-step options: softmax minus modal late-training steps, seeds 0–4: [-0.247, 0.713, -0.131, -0.105, 3.19]; mean change +0.684 (+3.35%).
- Recursive + pruning + hindsight: softmax minus modal late-training steps, seeds 0–4: [-0.856, 0.509, 0.304, 1.621, 1.783]; mean change +0.672 (+3.85%).

## Option behaviour

| Policy | Longest observed activation per seed | Hindsight updates, mean ± SD | Options pruned, mean ± SD |
| --- | --- | --- | --- |
| Two-step options | [2.0, 2.0, 2.0, 2.0, 2.0] | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Two-step options + softmax | [2.0, 2.0, 2.0, 2.0, 2.0] | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Recursive + pruning + hindsight | [7.0, 9.0, 9.0, 11.0, 7.0] | 1324.600 ± 276.145 | 6.600 ± 3.362 |
| Recursive + pruning + hindsight + softmax | [8.0, 7.0, 6.0, 7.0, 8.0] | 1414.000 ± 155.755 | 7.400 ± 1.517 |

Independent softmax snapshot reruns were collected for every seed. Final hindsight updates, pruning counts and longest observed activations agree with the benchmark. Snapshot collection does not sample actions. The final partial episode is not forcibly closed in the snapshot runner.

| Seed | Composite final policies | Maximum stored depth | Longest executed option | Stochastic state entries |
| --- | --- | --- | --- | --- |
| 0 | 11 | 4 | 8 | 4 |
| 1 | 13 | 4 | 7 | 3 |
| 2 | 9 | 3 | 6 | 6 |
| 3 | 11 | 4 | 7 | 3 |
| 4 | 10 | 4 | 8 | 7 |

### Seed 0 examples

Seed 0 was chosen before inspecting results, consistent with the Nine Rooms presentation. Final options below follow the existing ranking by observed execution length then selection count. Each policy was sampled 200 times from its initiation state with an independent fixed analysis RNG, without learning. These samples assess option targets, not overall task success; they do not alter training.

| Option | Start → target | Depth | Observed lengths (steps: count) | Sampled target success | Sampled mean length | Unique sampled action paths |
| --- | --- | --- | --- | --- | --- | --- |
| 22 | [1, 1] → [3, 7] | 3 | {'8': 132} | 100.0% | 8.00 | 2 |
| 27 | [3, 7] → [9, 9] | 4 | {'8': 70} | 100.0% | 8.00 | 1 |
| 21 | [4, 8] → [9, 9] | 3 | {'6': 137} | 100.0% | 6.00 | 1 |
| 23 | [3, 7] → [8, 8] | 3 | {'6': 136} | 100.0% | 6.00 | 1 |
| 24 | [2, 4] → [4, 8] | 3 | {'6': 104} | 100.0% | 6.00 | 1 |
| 26 | [2, 2] → [3, 7] | 3 | {'6': 26} | 100.0% | 6.00 | 1 |

## Interpretation and limits

At T=1, softmax did not improve average late-training performance in either paired comparison. Flat softmax was worse in two of five seeds and better in three; its mean was dominated by a large regression in seed 4. Full softmax was worse in four of five seeds. Both softmax variants retained an 18-step final evaluation path in seed 4; the other four seeds had optimal 16-step paths. Their modal counterparts had 16-step paths in every seed. All methods achieved 100% final success.

The full softmax method still produced useful multi-room skills: in seed 0, option 22 travels from (1,1) through the upper doorway to (3,7), while option 27 continues through the right doorway to the task goal (9,9). Each takes eight primitive steps. All six illustrated final options reached their own targets in all 200 sampled rollouts. Option 22 produced two sampled action paths; the other five produced one. These final skills were reliable in the sampled analysis, but did not yield a better overall learning result than the modal full method. Their reliability does not explain the training difference causally.

Softmax permits lower-frequency demonstrated actions, but raw-count softmax can become almost deterministic when count differences are large. Stored depth describes references to child options, not necessarily their execution. Read the measured duration distributions alongside hierarchy depth. Stochastic state entries count option/state tables with at least two probabilities above 1e-12; they are not distinct environment cells. Small probabilities may not appear in 200 samples.

Environment steps and hyperparameters are matched; exploration, update count, compute, and memory are not. Longer options change exploration per primitive step. Under softmax, hindsight supports any demonstrated action with nonzero numerical probability, rather than modal actions alone. Its replay is not importance weighted; a full-method difference cannot be attributed to sampling alone. Flat softmax provides a comparison without hindsight. These are five-seed results on one deterministic fixed-start task at one temperature, not a temperature sweep or generalization test. Sample SD describes variation; no significance claim is made.

## Artifacts and reproduction

- `learning_curves.png`: all eight freshly trained methods, training success and steps.
- `learning_summary.png`: primary paired policies and Q-learning, adding cumulative goals and late zoom.
- `option_sampling.png`: seed-0 duration distributions under actual stochastic policies.
- `episodes.csv`, `evaluation.csv`, `summary.csv`, `aggregate.json`: raw and aggregated benchmark data.
- `qualitative.json`: all-seed final option statistics.
- `../option_snapshots/composed_pruned_hindsight_softmax_T1/seed_N/`: HTML viewer, PNGs, raw snapshots, modal representative paths and `stochastic_rollouts.json`.

```powershell
python compare_agents.py --config configs/comparison_softmax.yaml
0..4 | ForEach-Object { python visualize_options.py --config configs/comparison_softmax.yaml --agent composed_pruned_hindsight_softmax --seed $_ }
python analyze_softmax.py
```

Existing historical results were preserved. Compare generated CSVs rather than timing across sessions; wall-clock speed is not an outcome of this experiment. Validation: 24 unit tests passed; all-seed snapshot/benchmark consistency, option references, sampled execution bounds and PNG outputs were checked. Sampling used 200 rollouts for each of 20 final options in each of five seeds (20,000 total).
