"""A small, discrete Four Rooms environment for reinforcement learning.

Install: pip install gymnasium

Observations are cell IDs, or joint position/goal IDs with observe_goal=True.
Actions: 0 = up, 1 = right, 2 = down, 3 = left.
"""

import gymnasium as gym
from gymnasium import spaces


class FourRoomsEnv(gym.Env):
    """An 11x11 grid divided into four rooms by walls with four doorways."""

    metadata = {"render_modes": ["human", "ansi"]}
    ROOMS_PER_SIDE = 2

    # Each action changes the agent's (row, column) position.
    MOVES = {
        0: (-1, 0),  # Up
        1: (0, 1),   # Right
        2: (1, 0),   # Down
        3: (0, -1),  # Left
    }

    def __init__(
        self,
        start=(1, 1),
        goal=(9, 9),
        max_steps=200,
        random_start=False,
        render_mode=None,
        random_goal=False,
        observe_goal=False,
    ):
        super().__init__()

        self.size = 6 * self.ROOMS_PER_SIDE - 1
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.random_start = random_start
        self.random_goal = random_goal
        self.observe_goal = observe_goal
        if random_goal and not observe_goal:
            raise ValueError('random_goal requires observe_goal=True to keep observations Markov')
        self.render_mode = render_mode

        if render_mode not in (None, "human", "ansi"):
            raise ValueError("render_mode must be None, 'human', or 'ansi'")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        # Five-cell rooms separated by walls with centered doorways.
        partitions = range(5, self.size, 6)
        centers = range(2, self.size, 6)
        self.walls = {(wall, col) for wall in partitions for col in range(self.size)}
        self.walls.update((row, wall) for wall in partitions for row in range(self.size))
        self.walls.difference_update((wall, center) for wall in partitions for center in centers)
        self.walls.difference_update((center, wall) for wall in partitions for center in centers)

        # Assign a unique integer state ID to every walkable cell.
        self.positions = [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if (row, col) not in self.walls
        ]
        self.position_to_state = {
            position: state for state, position in enumerate(self.positions)
        }

        if start not in self.position_to_state:
            raise ValueError("start must be a walkable (row, column) position")
        if goal not in self.position_to_state:
            raise ValueError("goal must be a walkable (row, column) position")
        if start == goal:
            raise ValueError("start and goal must be different")

        n_cells = len(self.positions)
        self.observation_space = spaces.Discrete(n_cells ** 2 if observe_goal else n_cells)
        self.action_space = spaces.Discrete(4)
        self.possible_starts = [p for p in self.positions if p != self.goal]

        self.agent_position = self.start
        self.steps = 0
        self.finished = False

    def reset(self, *, seed=None, options=None):
        """Start a new episode and return (state, info)."""
        super().reset(seed=seed)

        if self.random_goal:
            goals = self.positions if self.random_start else self.possible_starts_for(self.start)
            self.goal = goals[int(self.np_random.integers(len(goals)))]
        if self.random_start:
            starts = self.possible_starts_for(self.goal) if self.random_goal else self.possible_starts
            index = self.np_random.integers(len(starts))
            self.agent_position = starts[int(index)]
        else:
            self.agent_position = self.start

        self.steps = 0
        self.finished = False

        if self.render_mode == "human":
            self.render()

        return self.observation(), self.observation_info()

    def possible_starts_for(self, excluded):
        return [position for position in self.positions if position != excluded]

    def observation(self):
        """Joint tabular ID: position_cell * number_of_cells + goal_cell."""
        position = self.position_to_state[self.agent_position]
        if self.observe_goal:
            return position * len(self.positions) + self.position_to_state[self.goal]
        return position

    def observation_info(self):
        if self.observe_goal:
            return {'agent_position': self.agent_position, 'goal_position': self.goal}
        return {}

    def decode_observation(self, state):
        """Decode a joint observation into (agent_position, goal_position)."""
        if not self.observe_goal:
            raise ValueError('decode_observation requires observe_goal=True')
        position, goal = divmod(int(state), len(self.positions))
        return self.positions[position], self.positions[goal]

    def step(self, action):
        """Move once; return (state, reward, terminated, truncated, info)."""
        if self.finished:
            raise RuntimeError("Episode finished. Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError("Action must be an integer from 0 to 3")

        dr, dc = self.MOVES[int(action)]
        row, col = self.agent_position
        next_position = (row + dr, col + dc)

        # Bumping into a wall (or leaving the grid) keeps us in place.
        if next_position in self.position_to_state:
            self.agent_position = next_position

        self.steps += 1
        terminated = self.agent_position == self.goal
        truncated = self.steps >= self.max_steps and not terminated
        self.finished = terminated or truncated
        reward = 1.0 if terminated else 0.0

        if self.render_mode == "human":
            self.render()

        return self.observation(), reward, terminated, truncated, self.observation_info()

    def render(self):
        """Show the grid: A = agent, G = goal, # = wall, . = free cell."""
        lines = []
        for row in range(self.size):
            symbols = []
            for col in range(self.size):
                position = (row, col)
                if position == self.agent_position:
                    symbol = "A"
                elif position == self.goal:
                    symbol = "G"
                elif position in self.walls:
                    symbol = "#"
                else:
                    symbol = "."
                symbols.append(symbol)
            lines.append(" ".join(symbols))

        grid = "\n".join(lines)
        if self.render_mode == "human":
            print(grid, end="\n\n")
        return grid
