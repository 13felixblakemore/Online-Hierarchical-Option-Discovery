import copy
import unittest
import numpy as np
from agents import Transition
from deep_agents import DQNAgent, OutcomeDQNAgent, NeuralOption
from nine_rooms import NineRoomsEnv
from experiment import make_agent, update_agent


class DeepTests(unittest.TestCase):
    def setUp(self):
        self.env = NineRoomsEnv(observe_goal=True, random_start=True, random_goal=True)

    def state(self, xy, goal=(15, 15)):
        n = len(self.env.positions)
        return self.env.position_to_state[xy]*n+self.env.position_to_state[goal]

    def test_outcomes_pool_positions_and_goals(self):
        a = OutcomeDQNAgent(self.env, seed=1, min_occurrences=2, bc_epochs=5)
        for row, goal in [(1, (15, 15)), (8, (0, 0))]:
            s = [self.state((row, c), goal) for c in (0, 1, 2)]
            a.successful_trajectories.append((Transition(s[0], 1, 0, s[1]), Transition(s[1], 1, 0, s[2])))
        a.discover_options()
        self.assertEqual(len(a.options), 1)
        o = a.options[0]
        self.assertEqual(o.outcome, (0, 2))
        self.assertEqual(o.demonstrations, 2)
        self.assertTrue(o.can_initiate(self.state((16, 16))))
        stack = [dict(index=0, steps=0, decisions=0, origin=np.array([8, 0]))]
        a._advance_option_stack(stack, self.state((8, 2)))
        self.assertFalse(stack)

    def test_composition_and_har_duration(self):
        a = OutcomeDQNAgent(self.env, seed=2, bc_epochs=5)
        s = [self.state((1, c)) for c in (0, 1, 2, 3, 4)]
        seg = (Transition(s[0], 1, 0, s[1]), Transition(s[1], 1, 0, s[2]))
        child = NeuralOption((0, 2), [seg], a, 1)
        for p in child.network.p:
            p.fill(0)
        child.network.p[-1][1] = 1
        a.options.append(child)
        parent = NeuralOption((0, 4), [(Transition(s[0], 4, 0, s[2], 2), Transition(s[2], 4, 1, s[4], 2, True))], a, 1)
        for p in parent.network.p:
            p.fill(0)
        parent.network.p[-1][4] = 1
        a.options.append(parent)
        self.assertEqual(parent.depth, 2)
        trajectory = tuple(Transition(s[i], 1, float(i == 3), s[i+1], terminated=i == 3) for i in range(4))
        match = a._match_hindsight(1, trajectory, 0)
        self.assertEqual(match[1:], (4, s[4], True))
        self.assertAlmostEqual(match[0], a.gamma**3)
        bad = (Transition(s[0], 0, 0, s[1]),)+trajectory[1:]
        self.assertIsNone(a._match_hindsight(1, bad, 0))

    def test_target_sync_and_frozen_evaluation(self):
        a = DQNAgent(self.env, seed=1, batch_size=2, train_every=1, target_interval=1)
        s, _ = self.env.reset(seed=1)
        for _ in range(3):
            a.update(s, 1, 1, s, True)
        self.assertGreater(a.updates, 0)
        for p, q in zip(a.network.p, a.target.p):
            np.testing.assert_array_equal(p, q)
        saved = copy.deepcopy(a.network.p)
        a.learning_enabled = False
        a.update(s, 1, 0, s, False)
        for p, q in zip(saved, a.network.p):
            np.testing.assert_array_equal(p, q)

    def test_factory_and_episode_boundaries(self):
        for kind in ('dqn', 'outcome_dqn'):
            a = make_agent(dict(type=kind), self.env, 0)
            s, _ = self.env.reset(seed=1)
            action = a.select_action(s)
            ns, r, done, _, _ = self.env.step(action)
            update_agent(a, s, action, r, ns, done, True)
            if kind == 'outcome_dqn':
                self.assertIsNone(a._choice)
                self.assertFalse(a._stack)


if __name__ == '__main__':
    unittest.main()
