import taichi as ti
from typing import Union, List
from .forces.force import Force

@ti.data_oriented
class System(object):
    def __init__(
            self,
            n_atoms: int=1,
            forces: Union[None, List[Force]]=None,
        ):
        self.n_atoms = n_atoms
        self.forces = forces
        
