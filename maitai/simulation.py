import taichi as ti

@ti.data_oriented
class Simulation(object):
    def __init__(self, system, integrator) -> None:
        self.system = system
        self.integrator = integrator
        self._energy = ti.field(float)
        self._geometry = ti.field(float, shape=(system.n_atoms, 3))
        self._grad = ti.field(float, shape=(system.n_atoms, 3))
        self._velocity = ti.field(float, shape=(system.n_atoms, 3))

    def set_geometry(self, geometry: float) -> None:
        if isinstance(geometry, ti.Field):
            self._geometry.copy_from(geometry)
        elif "numpy" in type(geometry).lower():
            self._geometry.from_numpy(np.ndarray)
        elif "torch" in type(geometry).lower():
            self._geometry.from_torch(geometry)

    def get_geometry(self, geometry: float) -> None:
        return self._geometry

    @ti.kernel
    def get_energy(self) -> float:
        energy = 0.0
        for force in ti.static(self.system.forces):
            energy = energy + force.get_energy(self._geometry)
        self._energy = energy
        return self._energy

    @ti.func
    def zero_grad(self):
        self._grad.fill(0.0)

    @ti.func
    def get_grad(self) -> ti.f32:
        self.zero_grad()
        for force in ti.static(self.system.forces):
            force.get_grad(self._geometry, self._grad)
        return self._grad
