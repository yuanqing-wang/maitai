import taichi as ti
import abc

@ti.data_oriented
class Force(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_energy(self, geometry: float) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def get_grad(self, geometry: float, grad: float) -> None:
        raise NotImplementedError
