from collections import deque
import unittest

from agents import RepetitionAgent, Transition
from compare_agents import run
from experiment import make_agent, make_env
from four_rooms import FourRoomsEnv


class RandomGoalTests(unittest.TestCase):
    def test_goal_is_observed_and_both_endpoints_vary_reproducibly(self):
        env = FourRoomsEnv(random_start=True, random_goal=True, observe_goal=True)
        other = FourRoomsEnv(random_start=True, random_goal=True, observe_goal=True)
        self.assertEqual(env.observation_space.n, 104 * 104)
        tasks = []
        for episode in range(100):
            state, info = env.reset(seed=42 if episode == 0 else None)
            self.assertEqual((state, info), other.reset(seed=42 if episode == 0 else None))
            position, goal = env.decode_observation(state)
            self.assertNotEqual(position, goal)
            self.assertEqual(info, {'agent_position': position, 'goal_position': goal})
            tasks.append((position, goal))
        self.assertGreater(len({p for p, g in tasks}), 40)
        self.assertGreater(len({g for p, g in tasks}), 40)
        env.agent_position = (1, 1)
        env.goal = (1, 2)
        first = env.observation()
        env.goal = (9, 9)
        self.assertNotEqual(first, env.observation())
        with self.assertRaises(ValueError):
            FourRoomsEnv(random_goal=True)

    def test_goal_stays_constant_and_random_task_terminates_at_its_goal(self):
        env = FourRoomsEnv(random_start=True, random_goal=True, observe_goal=True)
        state, _ = env.reset(seed=12)
        goal = env.goal
        paths = {env.agent_position: []}
        queue = deque(paths)
        while queue:
            position = queue.popleft()
            for action, (dr, dc) in env.MOVES.items():
                next_position = position[0] + dr, position[1] + dc
                if next_position in env.position_to_state and next_position not in paths:
                    paths[next_position] = paths[position] + [action]
                    queue.append(next_position)
        for i, action in enumerate(paths[goal]):
            state, reward, terminal, truncated, _ = env.step(action)
            self.assertEqual(env.decode_observation(state), (env.agent_position, goal))
            self.assertEqual(terminal, i == len(paths[goal])-1)
            self.assertEqual(reward, float(terminal))
            self.assertFalse(truncated)

    def test_factories_and_runner_pair_tasks_for_both_methods(self):
        config = dict(environment=dict(random_start=True, random_goal=True, observe_goal=True,
                                       max_steps=2), agent={},
                      experiment=dict(environment_steps=20, evaluation_episodes=5))
        histories = [run(config, kind, 0) for kind in ('q_learning', 'composed_pruned_hindsight')]
        for index in (0, 1):
            tasks = [[(row['start_cell'], row['goal_cell']) for row in result[index]]
                     for result in histories]
            self.assertEqual(*tasks)
        env = make_env(config['environment'])
        for kind in ('q_learning', 'repetition'):
            agent = make_agent({'type': kind}, env, 0)
            self.assertEqual(agent.q.shape, (10816, 4))

    def test_hindsight_does_not_mix_task_goals(self):
        agent = RepetitionAgent(16, 1, min_occurrences=1, manager_hindsight=True,
                                compose_options=True)
        path = (Transition(1, 0, 0, 5), Transition(5, 0, 1, 9, terminated=True))
        agent.successful_trajectories.append(path)
        agent.successful_primitive_trajectories.append(path)
        agent.discover_options()
        self.assertIsNotNone(agent._match_hindsight(0, path, 0))
        changed_goal = tuple(Transition(t.state+1, t.action, t.reward, t.next_state+1,
                                        terminated=t.terminated) for t in path)
        self.assertIsNone(agent._match_hindsight(0, changed_goal, 0))
