import taichi as ti
from .integrator import Integrator

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
        geometry: ti.f32,
        velocity: ti.f32,
        acceleration: ti.f32,
    ):
        velocity += acceleration * timestep
        geometry += velocity * timestep
