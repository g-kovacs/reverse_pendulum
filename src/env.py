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
        self.maxX = 600
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
        self._action_space = spaces.Discrete(7) #spaces.Box(-self.maxT, self.maxT, shape=(1,))
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
        tourque = np.linspace(-self.maxT, self.maxT, self.action_space.n)[action]
        F = (2.0*tourque - self.coefR*self.mTot*self.g/2.0)/self.radW

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
        return np.array(dt_replace(self.state).flatten()), action, 1.0, terminate

    def reset(self):
        self.state = State(p_dG=0.01)
        return np.array(dt_replace(self.state).flatten())

    def render(self, mode='human'):
        screen_w = 600
        screen_h = 400

        world_width = self.maxX * 1.5
        scale = screen_w/world_width
        car_top = 100  # TOP OF CART
        pole_w = 10.0
        polelen = scale * self.lenP
        car_w = 60.0
        car_h = 30.0
        wheel_r = 4.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_w, screen_h)

            # Wheel geometry
            wheel = rendering.make_circle(wheel_r)
            self.wheeltrans_l = rendering.Transform(
                translation=(-1/3*car_w, wheel_r))
            wheel.set_color(0.7, 0, 0)
            wheel.add_attr(self.wheeltrans_l)
            self.viewer.add_geom(wheel)
            wheel = rendering.make_circle(wheel_r)
            self.wheeltrans_r = rendering.Transform(
                translation=(1/3*car_w, wheel_r))
            wheel.set_color(0.7, 0, 0)
            wheel.add_attr(self.wheeltrans_r)
            self.viewer.add_geom(wheel)

            # Car geometry
            l, r, t, b = -car_w / 2, car_w / 2, car_h / 2, -car_h / 2
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cartrans = rendering.Transform(translation=(0, wheel_r))
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)

            # Pole geometry
            l, r, t, b = -pole_w / 2, pole_w / 2, polelen, 0
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.7, .6, .3)
            self.poletrans = rendering.Transform(translation=(0, car_h))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.cartrans)
            self.viewer.add_geom(pole)

            # Axle
            axle = rendering.make_circle(pole_w/2)
            axle.add_attr(self.poletrans)
            axle.add_attr(self.cartrans)
            axle.set_color(.5, .5, .8)
            self.viewer.add_geom(axle)

            # Ground
            ground = rendering.Line((0, car_top), (screen_w, car_top))
            ground.set_color(0, 0, 0)
            self.viewer.add_geom(ground)

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
        self.cartrans.set_translation(carX, car_top + wheel_r)
        self.poletrans.set_rotation(-s.p_G)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

env = DoubleCartPoleEnv()
env.render()
input("asdfs")