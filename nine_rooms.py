"""Nine Rooms with the same movement, rewards and room dimensions as Four Rooms."""

from four_rooms import FourRoomsEnv


class NineRoomsEnv(FourRoomsEnv):
    """A 17x17 grid of nine 5x5 rooms connected by twelve doorways.

    Actions, integer observations, sparse goal reward, rendering, random starts,
    and time-limit handling are inherited from FourRoomsEnv.
    """

    ROOMS_PER_SIDE = 3

    def __init__(self, start=(1, 1), goal=(15, 15), max_steps=500,
                 random_start=False, render_mode=None, random_goal=False,
                 observe_goal=False):
        super().__init__(start=start, goal=goal, max_steps=max_steps,
                         random_start=random_start, render_mode=render_mode,
                         random_goal=random_goal, observe_goal=observe_goal)
