import unittest

from agents import RepetitionAgent, StatePairDiscovery, Transition
from four_rooms import FourRoomsEnv
from experiment import make_agent
from train import train


class RepetitionAgentTests(unittest.TestCase):
    def test_hindsight_is_opt_in_and_uses_terminal_discounted_return(self):
        for enabled in (False, True):
            agent = RepetitionAgent(3, 1, alpha=1, gamma=0.5, min_occurrences=1,
                                    manager_hindsight=enabled)
            path = (Transition(0, 0, 2, 1), Transition(1, 0, 4, 2, terminated=True))
            agent.successful_trajectories.append(path)
            agent.successful_primitive_trajectories.append(path)
            agent.q[2, 0] = 100
            rng_state = agent.rng.bit_generator.state
            agent.discover_options()
            self.assertEqual(agent.q[0, 1], 4 if enabled else 0)
            self.assertEqual(agent.hindsight_updates, int(enabled))
            self.assertEqual(agent.q[2, 0], 100)
            self.assertEqual(agent.environment_steps, 0)
            self.assertEqual(agent.options[0].selections, 0)
            self.assertEqual(agent.options[0].last_used_step, 0)
            self.assertEqual(agent.rng.bit_generator.state, rng_state)

    def test_hindsight_bootstraps_nonterminal_option_termination(self):
        agent = RepetitionAgent(4, 1, alpha=1, gamma=0.5, min_occurrences=1,
                                manager_hindsight=True)
        prefix = (Transition(0, 0, 2, 1), Transition(1, 0, 4, 2))
        agent.successful_trajectories.append(prefix)
        agent.successful_primitive_trajectories.append(
            prefix + (Transition(2, 0, 10, 3, terminated=True),))
        agent.q[2, 0] = 8
        agent.discover_options()
        self.assertEqual(agent.q[0, 1], 6)

    def test_hindsight_rejects_nonmodal_demonstrations_and_limits_updates(self):
        agent = RepetitionAgent(4, 2, alpha=1, min_occurrences=1,
                                manager_hindsight=True, hindsight_replay_limit=1)
        path = (Transition(0, 0, 0, 1), Transition(1, 0, 1, 2, terminated=True))
        agent.successful_trajectories.append(path)
        agent.successful_primitive_trajectories.extend([
            path, path, (Transition(0, 1, 0, 3), Transition(3, 0, 1, 2, terminated=True))])
        agent.discover_options()
        self.assertEqual(agent.hindsight_updates, 1)
        self.assertEqual(agent._match_hindsight(0, agent.successful_primitive_trajectories[-1], 0), None)

    def test_hindsight_follows_nested_execution_and_early_termination(self):
        agent = RepetitionAgent(5, 1, alpha=1, gamma=0.5, compose_options=True,
                                manager_hindsight=True, min_occurrences=1)
        for path in [
            (Transition(0, 0, 0, 1), Transition(1, 0, 0, 2)),
            (Transition(0, 1, 0, 2, 2), Transition(2, 0, 0, 3)),
        ]:
            agent.successful_trajectories.clear()
            agent.successful_trajectories.append(path)
            agent.discover_options()
        raw = tuple(Transition(s, 0, 1, s+1, terminated=s==3) for s in range(4))
        agent.successful_primitive_trajectories.append(raw)
        agent.q[3, 0] = 8
        self.assertEqual(agent.replay_manager_hindsight(1), 1)
        self.assertEqual(agent.q[0, 2], 2.75)  # 1+.5+.25 + .125*8
        # The parent can end while its child is running; replay must agree.
        agent.options[1].goal_state = 1
        self.assertEqual(agent._match_hindsight(1, raw, 0), (1, 1, 1, False))

    def test_pruning_preserves_dependencies_and_remaps_policy_q_and_replay(self):
        agent = RepetitionAgent(6, 1, compose_options=True, prune_options=True,
                                prune_after_steps=10, min_occurrences=1,
                                discovery_interval=100, epsilon=0)
        for path in [
            (Transition(0, 0, 0, 1), Transition(1, 0, 0, 2)),
            (Transition(2, 0, 0, 3), Transition(3, 0, 0, 4)),
            (Transition(2, 2, 0, 4, 2), Transition(4, 0, 0, 5)),
        ]:
            agent.successful_trajectories.clear()
            agent.successful_trajectories.append(path)
            agent.discover_options()
        agent.environment_steps = 100
        agent.options[2].last_used_step = 99
        agent.q[:, 2] = 7
        agent.q[:, 3] = 10
        agent.successful_trajectories.append((
            Transition(0, 0, 0, 1), Transition(1, 1, 0, 2, 2),
            Transition(2, 0, 0, 3)))
        removed = agent.prune_inactive_options()
        self.assertEqual(removed, [1])
        self.assertEqual([o.option_id for o in agent.options], [2, 3])
        self.assertEqual(agent.options[1].children, [0])
        self.assertIn(1, agent.options[1].action_counts[2])
        self.assertEqual(agent.q.shape, (6, 3))
        self.assertEqual(agent.q[2, 1], 7)
        self.assertEqual(agent.q[2, 2], 10)
        self.assertEqual(agent.successful_trajectories[-2][0].state, 0)
        self.assertEqual(agent.successful_trajectories[-1][0].state, 2)
        self.assertEqual(len(agent.successful_trajectories[-2]), 1)
        for state in range(2, 5):
            self.assertEqual(agent.select_action(state), 0)
            agent.update(state, 0, 0, state+1, state == 4)
        self.assertEqual(agent.options[0].last_used_step, 102)
        self.assertEqual(agent.options[0].selections, 1)
        self.assertEqual(agent.options[1].execution_lengths[3], 1)

    def test_pruning_grace_period_capacity_and_stable_ids(self):
        agent = RepetitionAgent(3, 1, min_occurrences=1, max_options=1,
                                prune_after_steps=10)
        path = (Transition(0, 0, 0, 1), Transition(1, 0, 0, 2))
        agent.successful_trajectories.append(path)
        agent.discover_options()
        agent.environment_steps = 9
        self.assertEqual(agent.prune_inactive_options(), [])
        agent.environment_steps = 10
        self.assertEqual(agent.prune_inactive_options(), [1])
        self.assertEqual(agent.q.shape, (3, 1))
        agent.discover_options()
        self.assertEqual(agent.options[0].option_id, 2)
        self.assertEqual(agent.options[0].created_step, 10)
        agent.select_action(0)
        with self.assertRaises(RuntimeError):
            agent.prune_inactive_options()

    def test_pruning_is_opt_in_at_episode_boundaries(self):
        for enabled in (False, True):
            agent = RepetitionAgent(4, 1, min_occurrences=1, prune_options=enabled,
                                    prune_after_steps=1, discovery_interval=1)
            agent.successful_trajectories.append((
                Transition(0, 0, 0, 1), Transition(1, 0, 0, 2)))
            agent.discover_options()
            agent.select_action(3)
            agent.update(3, 0, 0, 3, False, truncated=True)
            self.assertEqual(bool(agent.pruning_history), enabled)

    def test_recursive_option_plus_primitive_and_nested_execution(self):
        agent = RepetitionAgent(5, 1, compose_options=True, min_occurrences=1,
                                epsilon=0, alpha=1, gamma=0.5)
        paths = [
            (Transition(0, 0, 0, 1), Transition(1, 0, 0, 2)),
            (Transition(0, 1, 0, 2, 2), Transition(2, 0, 0, 3)),
            (Transition(0, 2, 0, 3, 3), Transition(3, 0, 0, 4)),
        ]
        for path in paths:
            agent.successful_trajectories.clear()
            agent.successful_trajectories.append(path)
            agent.discover_options()
        self.assertEqual([o.depth for o in agent.options], [1, 2, 3])
        self.assertEqual(agent.options[2].children, [1])
        agent.q[0, 3] = 10
        for state in range(4):
            self.assertEqual(agent.select_action(state), 0)
            agent.update(state, 0, 1, state+1, state == 3)
            if state < 3:
                self.assertEqual(agent._choice, 3)
        self.assertAlmostEqual(agent.q[0, 3], 1.875)
        self.assertEqual(agent._stack, [])
        self.assertEqual(agent.options[2].execution_lengths[4], 1)
        self.assertEqual(agent.successful_trajectories[-1][0].duration, 4)
        self.assertEqual(agent.successful_trajectories[-1][0].action, 3)

    def test_composition_of_two_options_and_truncation(self):
        agent = RepetitionAgent(5, 1, compose_options=True, min_occurrences=1,
                                epsilon=0, discovery_interval=100)
        for path in [
            (Transition(0, 0, 0, 1), Transition(1, 0, 0, 2)),
            (Transition(2, 0, 0, 3), Transition(3, 0, 0, 4)),
            (Transition(0, 1, 0, 2, 2), Transition(2, 2, 0, 4, 2)),
        ]:
            agent.successful_trajectories.clear()
            agent.successful_trajectories.append(path)
            agent.discover_options()
        self.assertEqual(agent.options[2].children, [0, 1])
        agent.q[0, 3] = 10
        for state in range(4):
            agent.select_action(state)
            agent.update(state, 0, 0, state+1, False)
        self.assertIsNone(agent._choice)
        self.assertEqual(agent.options[2].execution_lengths[4], 1)
        agent.select_action(0)
        agent.update(0, 0, 0, 1, False, truncated=True)
        self.assertEqual(agent._stack, [])
        self.assertIsNone(agent._choice)

    def test_groups_different_paths_by_endpoints(self):
        paths = [
            (Transition(0, 0, 0, 1), Transition(1, 1, 1, 3)),
            (Transition(0, 1, 0, 2), Transition(2, 0, 1, 3)),
        ]
        candidates = StatePairDiscovery(2).discover(paths)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0].segments), 2)
        agent = RepetitionAgent(4, 2, min_occurrences=2)
        agent.successful_trajectories.extend(paths)
        agent.discover_options()
        option = agent.options[0]
        self.assertEqual(option.select_action(1, agent.rng), 1)
        self.assertEqual(option.select_action(2, agent.rng), 0)
        agent.discover_options()
        self.assertEqual(len(agent.options), 1)

    def test_option_return_and_terminal_bootstrap(self):
        agent = RepetitionAgent(3, 1, alpha=1, gamma=0.5, epsilon=0,
                                min_occurrences=1)
        agent.successful_trajectories.append((
            Transition(0, 0, 0, 1), Transition(1, 0, 0, 2)))
        agent.discover_options()
        agent.q[0, 1] = 10
        agent.q[2, 0] = 100
        self.assertEqual(agent.select_action(0), 0)
        agent.update(0, 0, 2, 1, False)
        self.assertEqual(agent.select_action(1), 0)
        agent.update(1, 0, 4, 2, True)
        self.assertEqual(agent.q[0, 1], 4)
        self.assertIsNone(agent._choice)
        self.assertEqual(agent.episodes, 1)

    def test_truncation_bootstraps_without_recording_success(self):
        agent = RepetitionAgent(2, 1, alpha=1, gamma=0.5)
        agent.q[1, 0] = 6
        agent.select_action(0)
        agent.update(0, 0, 1, 1, False, truncated=True)
        self.assertEqual(agent.q[0, 0], 4)
        self.assertEqual(len(agent.successful_trajectories), 0)
        self.assertEqual(agent._trajectory, [])
        self.assertIsNone(agent._choice)

    def test_training_integration(self):
        env = FourRoomsEnv(start=(1, 1), goal=(1, 2), max_steps=20)
        try:
            agent = make_agent({"type": "repetition", "epsilon": 1,
                                "min_occurrences": 1,
                                "discovery_interval": 1}, env, 42)
            history = train(env, agent, 10, 42, 10)
            self.assertEqual(len(history), 10)
            self.assertEqual(agent.episodes, 10)
            self.assertGreater(len(agent.options), 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
