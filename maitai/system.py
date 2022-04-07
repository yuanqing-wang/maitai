import taichi as ti
from typing import Union, List
from .forces.force import Force

@ti.data_oriented
class System(object):
    def __init__(
            self,
            masses: Union[None, List[float]]=None,
            forces: Union[None, List[Force]]=None,
        ):
        self.masses = masses
        self.n_atoms = len(self.masses)
        self.forces = forces
