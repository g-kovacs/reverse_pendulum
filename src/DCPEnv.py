import gym
from gym import spaces
from dataclasses import dataclass, replace as dt_replace
import numpy as np


class DCPEnv(gym.Env):

    # Boundaries
    maxX = 6
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
    mC = car_width + 2 * (radW ** 2 ) * np.pi
    mP = lenP * 0.2
    p_I = 1/3 * mP * (lenP ** 2)
    mTot = mC + mP

    @dataclass
    class State:
        p_G: float = 0.0
        p_dG: float = 0.0
        c_X: float = 0.0
        c_dX: float = 0.0

        def flatten(self):
            return np.array([self.p_G, self.p_dG, self.c_X, self.c_dX])

        def wind_blow(self, torque):
            self.p_dG += torque / DCPEnv.p_I * DCPEnv.dt

        def add_torque(self, torque):
            F = (2.0*torque - DCPEnv.coefR * DCPEnv.mTot * DCPEnv.g/2.0)/DCPEnv.radW

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

    def __init__(self, timeStep=0.1):
        # spaces.Box(-self.maxT, self.maxT, shape=(1,))
        self._action_space = spaces.Discrete(7)
        boundary = np.array([self.maxG,
                             np.finfo(np.float32).max,
                             self.maxX,
                             np.finfo(np.float32).max],
                            dtype=np.float32)
        self._observation_space = spaces.Box(-boundary,
                                             boundary, dtype=np.float32)
        DCPEnv.dt = timeStep
        self.viewer = None

    def step(self, action):
        torque = np.linspace(-self.maxT, self.maxT,
                             self.action_space.n)[action]

        if np.random.random() < 1e-3:
            self.state.wind_blow(np.random.choice([-self.maxT * 10, self.maxT * 10]))

        self.state.add_torque(torque)

        terminate = False
        if np.abs(self.state.p_G) > self.maxG or np.abs(self.state.c_X) > self.maxX:
            terminate = True
        return np.array(dt_replace(self.state).flatten()), 1.0, terminate, {"action": action}

    def reset(self):
        self.state = DCPEnv.State(p_dG=0.01)
        return np.array(dt_replace(self.state).flatten())

    def test(self, model, render=True):
        obs, done, ep_reward = self.reset(), False, 0
        while not done:
            if render:
                self.render()
            action, _ = model.action_value(obs[None, :])
            obs, reward, done, _ = self.step(action)
            ep_reward += reward
        return ep_reward

    def render(self, mode='human'):
        screen_w = 600
        screen_h = 400

        world_width = self.maxX * 2
        scale = screen_w/world_width
        pole_w = scale * self.mP / self.lenP
        polelen = scale * self.lenP
        car_w = scale * self.car_width
        car_h = car_w / 2
        wheel_r = scale * self.radW

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_w, screen_h)

            # Car geometry
            l, r, t, b = -car_w / 2, car_w / 2, car_h, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cartrans = rendering.Transform(translation=(0, wheel_r))
            car.add_attr(self.cartrans)
            car.set_color(0.3, 0.3, 1)
            self.viewer.add_geom(car)

            # Wheel geometry
            wheel = rendering.make_circle(wheel_r)
            self.wheeltrans_l = rendering.Transform(
                translation=(-1/3*car_w, wheel_r))
            wheel.set_color(.6, .6, .6)
            wheel.add_attr(self.wheeltrans_l)
            self.viewer.add_geom(wheel)
            wheel = rendering.make_circle(wheel_r)
            self.wheeltrans_r = rendering.Transform(
                translation=(1/3*car_w, wheel_r))
            wheel.set_color(.6, .6, .6)
            wheel.add_attr(self.wheeltrans_r)
            self.viewer.add_geom(wheel)

            # Pole geometry
            l, r, t, b = -pole_w / 2, pole_w / 2, polelen, 0
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0, 0.9, 0.4)
            self.poletrans = rendering.Transform(translation=(0, car_h))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.cartrans)
            self.viewer.add_geom(pole)

            # Axle
            axle = rendering.make_circle(pole_w/2)
            axle.add_attr(self.poletrans)
            axle.add_attr(self.cartrans)
            axle.set_color(.45, .45, .45)
            self.viewer.add_geom(axle)

            self._pole_geom = pole

        if self.state is None:
            return None

        pole = self._pole_geom
        l, r, t, b = -pole_w / 2, pole_w / 2, polelen, 0
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        s = self.state
        carX = s.c_X * scale + screen_w / 2  # middle of cart
        self.wheeltrans_l.set_translation(carX - 1/3 * car_w, wheel_r)
        self.wheeltrans_r.set_translation(carX + 1/3 * car_w, wheel_r)
        self.cartrans.set_translation(carX, wheel_r)
        self.poletrans.set_rotation(-s.p_G)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def preview():
    env = DCPEnv()
    env.reset()
    env.render()
    input("")


if __name__ == "__main__":
    preview()
