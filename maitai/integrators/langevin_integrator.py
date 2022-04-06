from .integrator import Integrator

class LangevinIntegrator(Integrator):
    """ Integrates Langevin dynamics with BAOAB splitting.

    Parameters
    ----------
    temperature : float(default=298.0)
        Fictitious bath temperature.

    collision_rate : float(default=1.0)
        Collision rate.

    timestep : float(default=1.0)
        Time step.

    """
    def __init__(
        self,
        temperature: float=298.0,
        collision_rate: float=1.0,
        timestep: float=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.collision_rate = collision_rate
        self.timestep = timestep

    
