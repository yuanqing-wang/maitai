import abc
import taichi as ti

@ti.data_oriented
class Integrator(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def step(
            self,
            position: ti.f32,
            velocity: ti.f32,
            acceleration: ti.f32,
        ):
        return NotImplementedError
