"""Integration checks for the shared experiment entry points."""

import unittest

from compare_agents import run
from experiment import make_agent, make_env
from train import train


class ExperimentTests(unittest.TestCase):
    def test_budget_cutoff_is_partial_and_evaluation_preserves_training_counts(self):
        config = {
            'environment': {'start': [1, 1], 'goal': [9, 9], 'max_steps': 2},
            'agent': {'discovery_interval': 1, 'min_occurrences': 1},
            'experiment': {'environment_steps': 7, 'evaluation_episodes': 3},
        }
        for kind in ('q_learning', 'repetition', 'repetition_pruned', 'composed',
                     'composed_pruned', 'composed_pruned_hindsight', 'macro', 'macro_hindsight'):
            with self.subTest(kind=kind):
                rows, evaluation, summary = run(config, kind, seed=0)
                self.assertEqual([row['steps'] for row in rows], [2, 2, 2, 1])
                self.assertEqual([row['partial'] for row in rows], [0, 0, 0, 1])
                self.assertEqual(summary['completed_episodes'], 3)
                self.assertEqual(summary['options_created'], 0)
                self.assertEqual(summary['hindsight_updates'], 0)
                self.assertEqual(len(evaluation), 3)
                self.assertTrue(all(row['steps'] == 2 and not row['success']
                                    for row in evaluation))

    def test_random_baseline_trains_without_a_learning_update(self):
        env = make_env({'type': 'nine_rooms', 'max_steps': 2})
        try:
            agent = make_agent({'type': 'random'}, env, seed=0)
            history = train(env, agent, episodes=2, seed=0, log_every=2)
            self.assertEqual([row['steps'] for row in history], [2, 2])
            self.assertFalse(any(row['success'] for row in history))
        finally:
            env.close()
