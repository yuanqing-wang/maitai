import taichi as ti

@ti.data_oriented
class Simulation(object):
    def __init__(self, system, integrator) -> None:
        self.system = system
        self.integrator = integrator
        self._energy = ti.field(float)
        self._position = ti.field(float, shape=(system.n_atoms, 3))
        self._velocity = ti.field(float, shape=(system.n_atoms, 3))
        self._grad = ti.field(float, shape=(system.n_atoms, 3))

    def set_position(self, position: float) -> None:
        if isinstance(position, ti.Field):
            self._position.copy_from(position)
        elif "numpy" in type(position).lower():
            self._position.from_numpy(np.ndarray)
        elif "torch" in type(position).lower():
            self._position.from_torch(position)

    def get_position(self, position: float) -> None:
        return self._position

    @ti.kernel
    def get_energy(self) -> float:
        energy = 0.0
        for force in ti.static(self.system.forces):
            energy = energy + force.get_energy(self._position)
        self._energy = energy
        return self._energy

    @ti.func
    def zero_grad(self):
        self._grad.fill(0.0)

    @ti.func
    def get_grad(self) -> ti.f32:
        self.zero_grad()
        for force in ti.static(self.system.forces):
            force.get_grad(self._position, self._grad)
        return self._grad
