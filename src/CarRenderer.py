from gym.envs.classic_control import rendering
from dataclasses import dataclass
from numpy.random import uniform


@dataclass(init=False)
class Color:
    car: tuple
    wheel: tuple
    pole: tuple

    def __init__(self, c, p, w=(.6,) * 3):
        self.car = c
        self.pole = p
        self.wheel = w


class CarRenderer(rendering.Viewer):
    @dataclass
    class Colors:
        BLUE: Color = Color((.3, .3, 1, ), (0, .9, .4))
        RED: Color = Color((1, .3, .3), (.88, .29, .76))

        def RANDOM():
            return Color(uniform(size=3), uniform(size=3))

    def __init__(self, s_width=600, s_height=400, data_dict={}):
        self.dims = data_dict
        wW = self.dims.pop("wW")
        self.dims.update((x, y * s_width/wW)
                         for x, y in self.dims.items())
        self.dims['scale'] = s_width / wW
        self.cars = []
        self.width = s_width
        self.height = s_height
        super().__init__(s_width, s_height)

    def _genWheels(self, color: tuple):
        wheel = rendering.make_circle(self.dims['wR'])
        lTrans = rendering.Transform(
            translation=(-1/3*self.dims['cW'],
                         self.dims['wR']))
        wheel.set_color(*color)
        wheel.add_attr(lTrans)
        self.add_geom(wheel)
        wheel = rendering.make_circle(self.dims['wR'])
        rTrans = rendering.Transform(
            translation=(1/3*self.dims['cW'],
                         self.dims['wR']))
        wheel.set_color(*color)
        wheel.add_attr(rTrans)
        self.add_geom(wheel)
        return (lTrans, rTrans)

    def _genCar(self, color: tuple):
        cW = self.dims["cW"]
        cH = cW / 2
        l, r, t, b = -cW / 2, cW / 2, cH, 0
        car_geom = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        cartrans = rendering.Transform(translation=(0, self.dims['wR']))
        car_geom.add_attr(cartrans)
        car_geom.set_color(*color)
        self.add_geom(car_geom)
        return cartrans

    def _genPole(self, color: tuple, trans):
        pW = self.dims['pW']
        pL = self.dims['pL']
        l, r, t, b = -pW / 2, pW / 2, pL, 0
        pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        pole.set_color(*color)
        poletrans = rendering.Transform(translation=(0, self.dims['cW'] / 2))
        for t in [poletrans, trans]:
            pole.add_attr(t)
        self.add_geom(pole)
        self._poleGeom = pole
        return poletrans

    def _genAxle(self, trans: list):
        axle = rendering.make_circle(self.dims['pW']/2)
        axle.set_color(.45, .45, .45)
        for t in trans[::-1]:
            axle.add_attr(t)
        self.add_geom(axle)

    def add_car(self, color=None):
        if color is None:
            color = CarRenderer.Colors.RANDOM()
        car = []
        car.append(self._genCar(color.car))
        car.append(self._genPole(color.pole, car[0]))
        car.extend(self._genWheels(color.wheel))
        self._genAxle(car[:2])
        self.cars.append(car)

    def render_cars(self, states: list, mode="human"):
        scale = self.dims['scale']
        wR = self.dims['wR']
        cW = self.dims['cW']
        pW = self.dims['pW']
        pL = self.dims['pL']
        l, r, t, b = -pW / 2, pW / 2, pL, 0
        pole = self._poleGeom
        pole.v = [(l, b), (l, t), (r, t), (r, b)]
        for car, state in zip(self.cars, states):
            carX = state.c_X * scale + self.width / 2
            car[0].set_translation(carX, wR)
            car[1].set_rotation(-state.p_G)
            car[2].set_translation(carX - 1/3 * cW, wR)
            car[3].set_translation(carX + 1/3 * cW, wR)

        return self.render(return_rgb_array=mode == 'rgb_array')


if __name__ == "__main__":
    render_data = {"wW": 6 * 2, "pW": 0.48 / 2.4,
                                "pL": 2.4, "cW": 2, "wR": 0.2}
    r = CarRenderer(data_dict=render_data)
    r.add_car(CarRenderer.Colors.BLUE)
    input("")
