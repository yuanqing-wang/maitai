import abc
import taichi as ti
from simulation improt Simulation

@ti.data_oriented
class Integrator(object):
    def __init__(self) -> None:
        self.acceleration = ti.field()

    @ti.func
    @staticmethod
    def add(x, y):
        for idx0, idx1 in x:
            x[idx0, idx1] += y[idx0, idx1]
        return x

    @ti.func
    @staticmethod
    def scalar_multiply(x, y):
        for idx0, idx1 in x:
            x[idx0, idx1] *= y
        return x

    @abc.abstractmethod
    def step(
        self,
        simulation: Simulation,
    ):
        return NotImplementedError
