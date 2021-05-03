import gym
from gym.utils import seeding
from gym import error, spaces, utils
from dataclasses import dataclass, replace as dt_replace
import numpy as np
from copy import copy


@dataclass
class State:
    p_G: float = 0.0
    p_dG: float = 0.0
    c_X: float = 0.0
    c_dX: float = 0.0


@dataclass
class Action:
    torq: float


class DoubleCartPoleEnv(gym.Env):
    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def __init__(self, timeStep=0.1):
        # Boundaries
        self.maxX = 3
        self.maxG = 0.8
        self.maxT = 1

        # Cart data
        self.mC = 1
        self.mP = 0.1
        self.lenP = 0.8
        self.radW = 0.2
        self.friction = 1
        self.coefR = 0.001

        # Universal constants
        self.g = 9.81

        # Computed values
        self.p_I = 1/3 * self.mP * (self.lenP ** 2)
        self.mTot = self.mC + self.mP
        self.dt = timeStep
        self._action_space = spaces.Box(-self.maxT, self.maxT, shape=(1,))
        boundary = np.array([self.maxG*2,
                             np.finfo(np.float32).max,
                             self.maxX*2,
                             np.finfo(np.float32).max],
                            dtype=np.float32)
        self._observation_space = spaces.Box(-boundary,
                                             boundary, dtype=np.float32)
        
        self.reset()
        self.viewer = None

    def step(self, action):
        F = (2.0*action - self.coefR*self.mTot*self.g/2.0)/self.radW

        _a = (-1 * F - self.mP * 0.5 * self.lenP * (self.state.p_dG ** 2) *
              np.sin(self.state.p_G)) / self.mTot
        _b = (4/3 - self.mP * (np.cos(self.state.p_g) ** 2) / self.mTot)

        p_ddG = (self.g * np.sin(self.state.p_G) + np.cos(self.state.p_G) * _a) \
            / (0.5 * self.lenP * _b)

        _c = (self.state.p_dG ** 2) * np.sin(self.state.p_G) - \
            p_ddG * np.cos(self.state.p_G)
        c_ddX = (F + self.mP * 0.5 * self.lenP * _c) / self.mTot

        self.state.c_dX += self.dt * c_ddX
        self.state.c_X += self.dt * self.state.c_dX

        self.state.p_dG += self.dt * p_ddG
        self.state.p_G += self.dt * self.state.p_dG

        terminate = False
        if np.abs(self.state.p_G) > self.maxG or np.abs(self.state.c_X) > self.maxX:
            terminate = True
        return dt_replace(self.state), action, .0, terminate

    def reset(self):
        self.state = State(p_dG=0.1)

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
          self.viewer.close()
          self.viewer = None
