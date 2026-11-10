import unittest

import numpy as np

from agents import OptionCandidate, RepetitionAgent, TabularOption, Transition
from experiment import make_agent, make_env, variant_settings
from visualize_options import snapshot


class OptionPolicyTests(unittest.TestCase):
    def test_softmax_variant_preserves_matched_features(self):
        config = {'option_temperature': 1.0}
        for base in ('repetition', 'composed_pruned_hindsight'):
            modal = variant_settings(config, base)
            softmax = variant_settings(config, base + '_softmax')
            self.assertEqual(softmax.pop('option_action_selection'), 'softmax')
            self.assertEqual(modal.pop('option_action_selection'), 'most_frequent')
            self.assertEqual(modal, softmax)
        self.assertEqual(config, {'option_temperature': 1.0})

    def candidate(self):
        common = (Transition(0, 0, 0, 1), Transition(1, 0, 1, 2, terminated=True))
        rare = (Transition(0, 1, 0, 1), Transition(1, 0, 1, 2, terminated=True))
        return OptionCandidate(0, 2, (common, common, rare))

    def test_default_retains_modal_selection_and_seeded_ties(self):
        option = TabularOption(self.candidate())
        rng = np.random.default_rng(42)
        reference = np.random.default_rng(42)
        self.assertTrue(all(option.select_action(0, rng) == 0 for _ in range(20)))
        option.action_counts[0][1] = 2
        rng = np.random.default_rng(42)
        self.assertEqual([option.select_action(0, rng) for _ in range(30)],
                         [int(reference.choice([0, 1])) for _ in range(30)])

    def test_softmax_probabilities_sampling_and_temperature(self):
        option = TabularOption(self.candidate(), action_selection='softmax')
        probabilities = option.action_probabilities(0)
        self.assertAlmostEqual(probabilities[0], np.exp(2) / (np.exp(2) + np.exp(1)))
        rng = np.random.default_rng(42)
        reference = np.random.default_rng(42)
        actions = [option.select_action(0, rng) for _ in range(100)]
        self.assertEqual(actions, [int(reference.choice([0, 1], p=list(probabilities.values())))
                                   for _ in range(100)])
        self.assertEqual(set(actions), {0, 1})
        option.temperature = 0.1
        self.assertGreater(option.action_probabilities(0)[0], probabilities[0])
        option.temperature = 1000
        self.assertAlmostEqual(option.action_probabilities(0)[0], 0.5, places=3)
        self.assertEqual(option.action_probabilities(99), {})
        self.assertNotIn(2, probabilities)  # Undemonstrated actions are excluded.
        option.action_counts[0][0] = 1000000
        self.assertTrue(all(np.isfinite(p) for p in option.action_probabilities(0).values()))

    def test_invalid_settings_fail_before_discovery(self):
        for temperature in (0, -1, float('nan'), float('inf'), True, '1'):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                RepetitionAgent(3, 2, option_temperature=temperature)
        with self.assertRaises(ValueError):
            TabularOption(self.candidate(), action_selection='unknown')

    def test_factory_discovery_snapshot_and_hindsight_use_softmax(self):
        env = make_env({})
        try:
            agent = make_agent({'type': 'repetition', 'option_action_selection': 'softmax',
                                'option_temperature': 2, 'min_occurrences': 1}, env, 0)
            agent.successful_trajectories.extend(self.candidate().segments)
            agent.discover_options()
            option = agent.options[0]
            self.assertEqual(option.action_selection, 'softmax')
            self.assertEqual(option.temperature, 2)
            rare = self.candidate().segments[-1]
            rng_state = agent.rng.bit_generator.state
            self.assertIsNotNone(agent._match_hindsight(0, rare, 0))
            saved = snapshot(agent, 0)['options'][0]
            self.assertEqual(saved['action_selection'], 'softmax')
            self.assertEqual(saved['action_probabilities']['0'], option.action_probabilities(0))
            self.assertEqual(agent.rng.bit_generator.state, rng_state)
            option.action_selection = 'most_frequent'
            self.assertIsNone(agent._match_hindsight(0, rare, 0))
        finally:
            env.close()
