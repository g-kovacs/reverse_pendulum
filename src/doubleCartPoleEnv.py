import gym
from gym import spaces
from dataclasses import dataclass, replace as dt_replace
import numpy as np


@dataclass
class State:
    p_G: float = 0.0
    p_dG: float = 0.0
    c_X: float = 0.0
    c_dX: float = 0.0

    def flatten(self):
        return np.array([self.p_G, self.p_dG, self.c_X, self.c_dX])


class DoubleCartPoleEnv(gym.Env):
    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def __init__(self, timeStep=0.1):
        # Boundaries
        self.maxX = 2.4
        self.maxG = 12 * 2 * np.pi / 360
        self.maxT = 1

        # Cart data
        self.mC = 1
        self.mP = 0.1
        self.lenP = 1
        self.radW = 0.2
        self.friction = 1
        self.coefR = 0.001

        # Universal constants
        self.g = 9.81

        # Computed values
        self.p_I = 1/3 * self.mP * (self.lenP ** 2)
        self.mTot = self.mC + self.mP
        self.dt = timeStep
        # spaces.Box(-self.maxT, self.maxT, shape=(1,))
        self._action_space = spaces.Discrete(7)
        boundary = np.array([self.maxG,
                             np.finfo(np.float32).max,
                             self.maxX,
                             np.finfo(np.float32).max],
                            dtype=np.float32)
        self._observation_space = spaces.Box(-boundary,
                                             boundary, dtype=np.float32)

        self.viewer = None

    def step(self, action):
        torque = np.linspace(-self.maxT, self.maxT,
                              self.action_space.n)[action]
        F = (2.0*torque - self.coefR*self.mTot*self.g/2.0)/self.radW

        sinG = np.sin(self.state.p_G)
        cosG = np.cos(self.state.p_G)

        _a = (-1 * F - self.mP * 0.5 * self.lenP * (self.state.p_dG ** 2) *
              sinG) / self.mTot
        _b = (4/3 - self.mP * (cosG ** 2) / self.mTot)

        p_ddG = (self.g * sinG + cosG * _a) \
            / (0.5 * self.lenP * _b)

        _c = (self.state.p_dG ** 2) * sinG - \
            p_ddG * cosG
        c_ddX = (F + self.mP * 0.5 * self.lenP * _c) / self.mTot

        self.state.c_dX += self.dt * c_ddX
        self.state.c_X += self.dt * self.state.c_dX

        self.state.p_dG += self.dt * p_ddG
        self.state.p_G += self.dt * self.state.p_dG

        terminate = False
        if np.abs(self.state.p_G) > self.maxG or np.abs(self.state.c_X) > self.maxX:
            terminate = True
        return np.array(dt_replace(self.state).flatten()), 1.0, terminate, {"action": action}

    def reset(self):
        self.state = State(p_dG=0.01)
        return np.array(dt_replace(self.state).flatten())

    def render(self, mode='human'):
        screen_w = 600
        screen_h = 400

        world_width = self.maxX * 2
        scale = screen_w/world_width
        pole_w = 8.0
        polelen = scale * self.lenP
        car_w = 70.0
        car_h = 40.0
        wheel_r = 8.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_w, screen_h)

            # Wheel geometry
            wheel = rendering.make_circle(wheel_r)
            self.wheeltrans_l = rendering.Transform(
                translation=(-1/3*car_w, wheel_r))
            wheel.set_color(.7, .7, .7)
            wheel.add_attr(self.wheeltrans_l)
            self.viewer.add_geom(wheel)
            wheel = rendering.make_circle(wheel_r)
            self.wheeltrans_r = rendering.Transform(
                translation=(1/3*car_w, wheel_r))
            wheel.set_color(.7, .7, .7)
            wheel.add_attr(self.wheeltrans_r)
            self.viewer.add_geom(wheel)

            # Car geometry
            l, r, t, b = -car_w / 2, car_w / 2, car_h, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cartrans = rendering.Transform(translation=(0, wheel_r))
            car.add_attr(self.cartrans)
            car.set_color(0.3, 0.3, 1)
            self.viewer.add_geom(car)

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

env=DoubleCartPoleEnv()
env.reset()
env.render()
input("")