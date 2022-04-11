import taichi as ti

@ti.data_oriented
class Simulation(object):
    def __init__(self, system, integrator) -> None:
        self.system = system
        self.integrator = integrator
        self._energy = ti.field(ti.f32, shape=())
        self._position = ti.field(ti.f32, shape=(system.n_atoms, 3))
        self._velocity = ti.field(ti.f32, shape=(system.n_atoms, 3))
        self._grad = ti.field(ti.f32, shape=(system.n_atoms, 3))

    def set_position(self, position: float) -> None:
        if isinstance(position, ti.Field):
            self._position.copy_from(position)
        elif "numpy" in type(position).__name__.lower():
            self._position.from_numpy(np.ndarray)
        elif "torch" in type(position).__name__.lower():
            self._position.from_torch(position)

    def get_position(self) -> None:
        return self._position

    @ti.func
    def compute_energy(self):
        self._energy[None] = 0.0
        for force in ti.static(self.system.forces):
            force.get_energy(self._position, self._energy)

    def get_energy(self):
        return self._energy

    @ti.func
    def zero_grad(self):
        self._grad.fill(0.0)

    @ti.kernel
    def zero_velocity(self):
        for x, y in self._velocity:
            self._velocity[x, y] = 0.0

    @ti.func
    def compute_grad(self) -> ti.f32:
        self.zero_grad()
        for force in ti.static(self.system.forces):
            force.get_grad(self._position, self._grad)

    def get_grad(self):
        return self._grad

    @ti.func
    def compute_acceleration(self) -> ti.f32:
        masses = self.system.masses
        self.compute_grad()
        for idx_atom, idx_dimension in self._grad:
            self._grad[idx_atom, idx_dimension] = self._grad[idx_atom, idx_dimension]\
                / masses[idx_atom]
