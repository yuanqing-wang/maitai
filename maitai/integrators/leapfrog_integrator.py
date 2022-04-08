import taichi as ti
from .integrator import Integrator
from .simulation import Simulation

class LeapfrogIntegrator(Integrator):
    """ Implements Leapfrog integration.

    """
    def __init__(
        self,
        timestep: float=1.0,
    ):
        super().__init__()
        self.timestep = timestep

    @ti.func
    def step(
        self,
        simulation: Simulation,
    ):
    acceleration = simulation.get_acceleration()
    delta_v = self.scalar_multiply(acceleration, 0.5 * self.timestep)
    self.add(simulation._velocity, delta_v)
    delta_x = self.scalar_multiply(simulation._velocity, self.timestep)
    self.add(simulation._position, delta_x)
    acceleration = simulation.get_acceleration()
    delta_v = self.scalar_multiply(acceleration, 0.5 * self.timestep)
    self.add(simulation._velocity, delta_v)
