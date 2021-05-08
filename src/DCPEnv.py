import gym
from gym import spaces
from dataclasses import dataclass, astuple
import numpy as np
from CarRenderer import CarRenderer


class DCPEnv(gym.Env):

    # Boundaries
    carDist = 4
    maxG = 1
    maxT = 0.94

    # Cart data - !!! MODIFICATIONS HERE !!!
    car_width = 2
    lenP = 2.4
    radW = 0.2

    # Universal constants
    g = 9.81
    coefR = 0.001
    dt = 0.1

    # Computed values
    mC = car_width + 2 * (radW ** 2) * np.pi
    mP = lenP * 0.2
    p_I = 1/3 * mP * (lenP ** 2)
    mTot = mC + mP
    maxX = 6

    @dataclass()
    class State:
        p_G: float = 0.0
        p_dG: float = 0.0
        c_X: float = 0.0
        c_dX: float = 0.0

        def flatten(self):
            return (self.p_G, self.p_dG, self.c_X, self.c_dX)

        def wind_blow(self, torque):
            self.p_dG += torque / DCPEnv.p_I * DCPEnv.dt

        def noise(self):
            noise = np.random.standard_normal(4)
            scale = (.1, .01, .4, .1)
            self.p_G, self.p_dG, self.c_X, self.c_dX = (
                x + s * n for x, s, n in zip(self.flatten(), scale, noise))
            return self

        def add_torque(self, torque):
            F = (2.0*torque - DCPEnv.coefR *
                 DCPEnv.mTot * DCPEnv.g/2.0)/DCPEnv.radW

            sinG = np.sin(self.p_G)
            cosG = np.cos(self.p_G)

            _a = (-1 * F - DCPEnv.mP * 0.5 * DCPEnv.lenP * (self.p_dG ** 2) *
                  sinG) / DCPEnv.mTot
            _b = (4/3 - DCPEnv.mP * (cosG ** 2) / DCPEnv.mTot)

            p_ddG = (DCPEnv.g * sinG + cosG * _a) \
                / (0.5 * DCPEnv.lenP * _b)

            _c = (self.p_dG ** 2) * sinG - \
                p_ddG * cosG
            c_ddX = (F + DCPEnv.mP * 0.5 * DCPEnv.lenP * _c) / DCPEnv.mTot

            self.c_dX += DCPEnv.dt * c_ddX
            self.c_X += DCPEnv.dt * self.c_dX

            self.p_dG += DCPEnv.dt * p_ddG
            self.p_G += DCPEnv.dt * self.p_dG

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def __init__(self, numCars=1, timeStep=0.1):
        self._action_space = spaces.Discrete(7 * numCars)
        DCPEnv.maxX = (1 + numCars) / 2 * DCPEnv.carDist
        boundary = np.array([self.maxG,
                             np.finfo(np.float32).max,
                             self.maxX,
                             np.finfo(np.float32).max] * numCars,
                            dtype=np.float32)
        self._observation_space = spaces.Box(-boundary,
                                             boundary, dtype=np.float32)
        DCPEnv.dt = timeStep
        self.viewer = None
        self.render_data = {"wW": self.maxX * 2, "pW": self.mP / self.lenP,
                            "pL": self.lenP, "cW": self.car_width, "wR": self.radW}
        self.numCars = numCars

    def _init_renderer(self):
        viewer = CarRenderer(data_dict=self.render_data)
        for _ in range(self.numCars):
            viewer.add_car()
        return viewer

    def _convert_states(self):
        return np.array(sum((astuple(s) for s in self.states), tuple()))

    # ==================================================
    # =================== STEP =========================
    def step(self, action):
        torque = np.linspace(-self.maxT, self.maxT,
                             self.action_space.n)[action]

        if np.random.random() < 1e-4:
            t = np.random.standard_normal() * 0.2
            self.states[0].wind_blow(np.random.choice([-t, t]))
        self.states[0].add_torque(torque)

        terminate = False
        if np.abs(self.states[0].p_G) > self.maxG or np.abs(self.states[0].c_X) > self.maxX:
            terminate = True
        return np.array(self._convert_states()), 1.0, terminate, {"action": action}

    # ==================================================
    # =================== RESET ========================
    # ========== initialize cars here ==================
    def reset(self):
        self.states = [DCPEnv.State(
            c_X=DCPEnv.carDist*(1-self.numCars+2*i)/2).noise()
            for i in range(self.numCars)]
        return self._convert_states()

    def test(self, model, render=True):
        obs, done, ep_reward = self.reset(), False, 0
        model.reset_buffer(obs)
        while not done:
            if render:
                self.render()
            action, _ = model.action_value(obs[None, :], False)
            obs, reward, done, _ = self.step(action)
            ep_reward += reward
        return ep_reward

    def render(self):
        if self.viewer is None:
            self.viewer = self._init_renderer()

        if self.states is None:
            return None

        return self.viewer.render_cars(self.states)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def preview():
    env = DCPEnv(numCars=4)
    print(env.reset())
    env.render()
    input("")
    env.close()


if __name__ == "__main__":
    preview()
