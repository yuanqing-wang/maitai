import taichi as ti

@ti.data_oriented
class Simulation(object):
    def __init__(self, system, integrator) -> None:
        self.system = system
        self.integrator = integrator
        self._geometry = ti.field(float, shape=(system.n_atoms, 3), needs_grad=True)
        self._energy = ti.field(float)

    def set_geometry(self, geometry: float) -> None:
        self._geometry.copy_from(geometry)

    def get_geometry(self, geometry: float) -> None:
        return self._geometry

    @ti.kernel
    def evaluate(self) -> float:
        energy = 0.0
        for force in ti.static(self.system.forces):
            energy = energy + force.evaluate(self._geometry)
        self._energy = energy
        return energy

    @ti.kernel
    def grad(self):
        with ti.Tape(self._energy):
            self.evaluate()
