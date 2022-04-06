import taichi as ti

@ti.data_oriented
class Simulation(object):
    def __init__(self, system, integrator) -> None:
        self.system = system
        self.integrator = integrator
        self._geometry = ti.field(float, shape=(system.n_atoms, 3), needs_grad=True)
        self._get_energy = ti.field(float)

    def set_geometry(self, geometry: float) -> None:
        self._geometry.copy_from(geometry)

    def get_geometry(self, geometry: float) -> None:
        return self._geometry

    @ti.kernel
    def get_energy(self) -> float:
        get_energy = 0.0
        for force in ti.static(self.system.forces):
            get_energy = get_energy + force.get_energy(self._geometry)
        self._get_energy = get_energy
        return get_energy
