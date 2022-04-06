import taichi as ti
import abc

@ti.data_oriented
class Force(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def evaluate(self, geometry: float) -> float:
        raise NotImplementedError
