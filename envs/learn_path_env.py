import gym
from gym import spaces
import numpy as numpy

class LearningPathEnv(gym.Env):
    """
    Gridworld where each cell is a "module" with prereqs.
    Must visit prereqs before entering certain cells.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_shape=(4,4), start=(0,0), goal=(3,3), prerequisites=None):
        super().__init__()
        self.grid_shape = grid_shape
        self.start = start
        self.goal = goal
        # dict: cell → required prior cell, e.g. {(2,2):(1,1)}
        self.prerequisites = prerequisites or {}
        # action space: 0=up, 1=right,2=down,3=left
        self.action_space = spaces.Discrete(4)
        # observation: flattened index of current cell
        self.observation_space = spaces.Discrete(grid_shape[0]*grid_shape[1])
        self.reset()

    def reset(self):
        self.pos = self.start
        self.visited = {self.start}
        return self._get_obs()

    def step(self, action):
        r, c = self.pos
        if action == 0:
            r = max(r-1, 0)
        elif action == 1:
            c = min(c+1, self.grid_shape[1]-1)
        elif action == 2:
            r = min(r+1, self.grid_shape[0]-1)
        elif action == 3:
            c = max(c-1, 0)

        next_pos = (r, c)

        req = self.prerequisites.get(next_pos)

        if req and req not in self.visited:
            # blocked: stay in place
            reward = -5
            done = False
            next_pos = self.pos
        else:
            self.pos = next_pos
            self.visited.add(self.pos)
            if self.pos == self.goal:
                reward = +10
                done = True
            else:
                reward = -1
                done = False

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # flattens (r,c) to single int
        return self.pos[0]*self.grid_shape[1] + self.pos[1]

    def render(self, mode='human'):
        grid = np.full(self.grid_shape, '.')
        for req_target, req_source in self.prerequisites.items():
            grid[req_target] = 'P'
        grid[self.start] = 'S'
        grid[self.goal] = 'G'
        grid[self.pos] = 'X'
        print("\n".join(" ".join(row) for row in grid))
        print()