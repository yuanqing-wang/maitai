import abc
import taichi as ti
from simulation improt Simulation

@ti.data_oriented
class Integrator(object):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def step(self, simulation=Simulation):
        return NotImplementedError
