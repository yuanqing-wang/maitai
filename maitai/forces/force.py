import taichi as ti
import abc

@ti.data_oriented
class Force(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_energy(self, position: float) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def get_grad(self, position: float, grad: float) -> None:
        raise NotImplementedError
