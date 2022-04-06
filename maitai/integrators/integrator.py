import abc

class Integrator(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def step(self):
        return NotImplementedError
