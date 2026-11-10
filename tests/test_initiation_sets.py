import unittest
import json

import numpy as np

from analyze_initiation_sets import probe
from agents import OptionCandidate, RepetitionAgent, TabularOption, Transition
from experiment import variant_settings
from visualize_options import snapshot


class InitiationSetTests(unittest.TestCase):
    def test_saved_policy_probes_match_nested_execution_and_reject_missing_root_policy(self):
        agent = RepetitionAgent(16, 1, min_occurrences=1, compose_options=True,
                                epsilon=0, option_initiation='policy_support')
        for path in (
            (Transition(3, 0, 0, 7), Transition(7, 0, 0, 11)),
            (Transition(3, 1, 0, 11, 2), Transition(11, 0, 1, 15, terminated=True)),
        ):
            agent.successful_trajectories.clear()
            agent.successful_trajectories.append(path)
            agent.discover_options()
        data = json.loads(json.dumps(dict(n_cells=4, n_actions=1,
            positions=[(0, i) for i in range(4)], moves={0: (0, 1)},
            snapshot=snapshot(agent, 0))))
        self.assertEqual(probe(data, 1, 0, np.random.default_rng(0)),
                         dict(success=True, steps=3, reason='target'))
        self.assertEqual(probe(data, 1, 2, np.random.default_rng(0)),
                         dict(success=True, steps=1, reason='target'))
        # The child knows this cell, but the root has no action there.
        self.assertEqual(probe(data, 1, 1, np.random.default_rng(0)),
                         dict(success=False, steps=0, reason='undefined_policy'))
        agent.q[3, 2] = 10
        for state in (3, 7, 11):
            self.assertEqual(agent.select_action(state), 0)
            agent.update(state, 0, float(state == 11), state+4, state == 11)
        self.assertEqual(agent.options[1].execution_lengths[3], 1)

    def path(self):
        return (Transition(0, 0, 0, 1), Transition(1, 0, 1, 2, terminated=True))

    def test_default_is_single_state_and_broader_set_excludes_unknown_and_target(self):
        candidate = OptionCandidate(0, 2, (self.path(),))
        original = TabularOption(candidate)
        broad = TabularOption(candidate, initiation='policy_support')
        self.assertEqual(original.initiation_states, {0})
        self.assertEqual(broad.initiation_states, {0, 1})
        self.assertTrue(broad.can_initiate(1))
        self.assertFalse(broad.can_initiate(2))
        self.assertFalse(broad.can_initiate(3))

    def test_manager_executes_from_additional_start_and_hindsight_updates_that_row(self):
        agent = RepetitionAgent(4, 1, alpha=1, gamma=.5, epsilon=0,
                                min_occurrences=1, compose_options=True,
                                manager_hindsight=True, option_initiation='policy_support')
        agent.successful_trajectories.append(self.path())
        agent.successful_primitive_trajectories.append(self.path())
        agent.discover_options()
        self.assertEqual(agent.q[0, 1], .5)
        self.assertEqual(agent.q[1, 1], 1)
        self.assertEqual(agent.hindsight_updates, 2)
        self.assertEqual(agent.q[3, 1], 0)
        agent.q[1, 1] = 2
        self.assertEqual(agent._available_actions(1), [0, 1])
        self.assertEqual(agent._available_actions(3), [0])
        self.assertEqual(agent.select_action(1), 0)
        agent.update(1, 0, 1, 2, True)
        self.assertEqual(agent.options[0].execution_lengths[1], 1)
        self.assertIsNone(agent._choice)

    def test_goal_context_is_still_required(self):
        # Four cells per goal context: joint IDs position*4 + goal.
        path = (Transition(1, 0, 0, 5), Transition(5, 0, 1, 9, terminated=True))
        option = TabularOption(OptionCandidate(1, 9, (path,)), initiation='policy_support')
        self.assertTrue(option.can_initiate(5))
        self.assertFalse(option.can_initiate(6))  # Same position, different task goal.

    def test_variant_changes_only_initiation_and_rejects_invalid_mode(self):
        original = variant_settings({}, 'composed_pruned_hindsight')
        broad = variant_settings({}, 'composed_pruned_hindsight_support')
        self.assertEqual(original.pop('option_initiation'), 'single_state')
        self.assertEqual(broad.pop('option_initiation'), 'policy_support')
        self.assertEqual(original, broad)
        with self.assertRaises(ValueError):
            RepetitionAgent(3, 1, option_initiation='all')
