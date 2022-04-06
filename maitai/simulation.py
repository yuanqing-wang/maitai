import taichi as ti


@ti.data_oriented
class Simulation(object):
    def __init__(self, system, integrator) -> None:
        self.system = system
        self.integrator = integrator
