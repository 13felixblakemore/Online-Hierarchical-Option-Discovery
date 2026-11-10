from collections import deque
import unittest

from four_rooms import FourRoomsEnv
from nine_rooms import NineRoomsEnv
from experiment import make_env


class RoomEnvironmentTests(unittest.TestCase):
    def test_four_rooms_layout_is_unchanged(self):
        env = FourRoomsEnv()
        expected = {(5, c) for c in range(11)} | {(r, 5) for r in range(11)}
        expected -= {(5, 2), (5, 8), (2, 5), (8, 5)}
        self.assertEqual(env.walls, expected)
        self.assertEqual(env.observation_space.n, 104)

    def test_nine_rooms_layout_and_connectivity(self):
        env = NineRoomsEnv()
        self.assertEqual(env.size, 17)
        self.assertEqual(env.observation_space.n, 237)
        doors = {(w, c) for w in (5, 11) for c in (2, 8, 14)}
        doors |= {(c, w) for w in (5, 11) for c in (2, 8, 14)}
        self.assertEqual(len(doors), 12)
        remaining = set(env.positions)-doors
        rooms = []
        while remaining:
            stack = [remaining.pop()]
            count = 0
            while stack:
                row, col = stack.pop()
                count += 1
                for dr, dc in env.MOVES.values():
                    p = row+dr, col+dc
                    if p in remaining:
                        remaining.remove(p)
                        stack.append(p)
            rooms.append(count)
        self.assertEqual(sorted(rooms), [25]*9)
        # Find a shortest path and verify actual environment transitions/rewards.
        paths = {env.start: []}
        queue = deque([env.start])
        while queue:
            row, col = queue.popleft()
            for action, (dr, dc) in env.MOVES.items():
                p = row+dr, col+dc
                if p in env.position_to_state and p not in paths:
                    paths[p] = paths[(row, col)] + [action]
                    queue.append(p)
        self.assertEqual(len(paths), 237)
        self.assertEqual(len(paths[env.goal]), 28)
        env.reset(seed=0)
        for i, action in enumerate(paths[env.goal]):
            state, reward, terminated, truncated, _ = env.step(action)
            self.assertEqual(terminated, i == 27)
            self.assertEqual(reward, float(i == 27))
            self.assertFalse(truncated)
            self.assertTrue(env.observation_space.contains(state))

    def test_doors_collisions_time_limits_and_environment_factory(self):
        env = make_env(dict(type='nine_rooms', start=[4, 2], goal=[15, 15]))
        self.assertIsInstance(env, NineRoomsEnv)
        env.reset()
        env.step(2)
        self.assertEqual(env.agent_position, (5, 2))
        env.step(1)
        self.assertEqual(env.agent_position, (5, 2))
        env.step(2)
        self.assertEqual(env.agent_position, (6, 2))
        limited = NineRoomsEnv(max_steps=1)
        limited.reset()
        _, reward, terminated, truncated, _ = limited.step(0)
        self.assertEqual((reward, terminated, truncated), (0, False, True))
        with self.assertRaises(RuntimeError):
            limited.step(0)

    def test_random_start_is_seeded_and_never_the_goal(self):
        env = NineRoomsEnv(random_start=True)
        self.assertEqual(env.reset(seed=42)[0], env.reset(seed=42)[0])
        seen = set()
        for _ in range(100):
            env.reset()
            self.assertNotEqual(env.agent_position, env.goal)
            seen.add(env.agent_position)
        self.assertGreater(len(seen), 1)


if __name__ == '__main__':
    unittest.main()
