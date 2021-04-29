from dataclasses import dataclass

@dataclass
class Rotatable:
  rotation: float 
  angular_momentum: float

@dataclass
class Cart(Rotatable):
  position: float
  idx: int
  momentum: float
  wheel: Rotatable
  pendulum: Rotatable

@dataclass
class State:
  cart1: Cart

@dataclass
class Action:
  torque1: float

class Environment:
  # Cart data
  w_mass = 0.1
  c_mass = 1
  p_mass = 0.1
  p_length = 0.8
  w_radius = 0.2
  friction = 1

  # Universal constants
  g = 9.81

  # Computed values
  p_angular_mass = 1/3 * p_mass * (p_length ** 2)
  w_angular_mass = 1/2 * w_mass * (w_radius ** 2)
  total_mass = c_mass + 2*w_mass + p_mass

  def __init__(self, timeStep = 0.1, cartCount = 1):
    self.dt = timeStep
    self.cartCount = cartCount

  def step(self, state, action):
    state.cart1.wheel.angular_momentum += self.dt * action.torque1

    wheelForce = state.cart1.wheel.angular_momentum / w_radius
    pressingForce = totalMass / 2 * g
    rotatingForce = max(wheelForce - pressingForce * friction, 0)

    angularVelocity = rotatingForce * w_radius / w_angular_mass
    state.cart1.wheel.rotation += angularVelocity * self.dt

    state.cart1.momentum += 2 * rotatingForce
    cVelocity = state.cart1.momentum / c_mass
    state.cart1.position += self.dt * cVelocity