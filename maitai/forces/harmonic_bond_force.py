import taichi as ti
from .force import Force
from typing import Tuple

class HarmonicBondForce(Force):
    """ Implements an interaction between pairs of particles that varies
    harmonically with the distance between them.


    """
    def __init__(self, bond_parameters=[]):
        super().__init__()
        self.bond_parameters = bond_parameters

    def add_bond(
            self,
            particle1: int,
            particle2: int,
            length: float,
            k: float,
        ) -> None:
        """ Add a bond term to the force field. """
        self.bond_parameters.append(
            (particle1, particle2, length, k)
        )

    def get_bond_parameters(self, index: int) -> Tuple[int, int, float]:
        return self.bond_parameters[index]

    def get_num_bonds(self) -> int:
        return len(self.bond_parameters)

    def set_bond_parameters(
            self,
            index: int,
            particle1: int,
            particle2: int,
            length: float,
            k: float,
        ) -> None:

        self.bond_parameters[index] = (particle1, particle2, k)

    @ti.func
    def get_energy(self, position: float) -> float:
        energy = 0.0
        for particle1, particle2, length, k in ti.static(self.bond_parameters):
            x1, y1, z1 = position[particle1, 0], position[particle1, 1], position[particle1, 2]
            x2, y2, z2 = position[particle2, 0], position[particle2, 1], position[particle2, 2]
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
            energy = energy + 0.5 * k * (distance - length) ** 2
        return energy

    @ti.func
    def get_grad(self, position: float, grad: float) -> None:
        for particle1, particle2, length, k in ti.static(self.bond_parameters):
            x1, y1, z1 = position[particle1, 0], position[particle1, 1], position[particle1, 2]
            x2, y2, z2 = position[particle2, 0], position[particle2, 1], position[particle2, 2]
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
            d_energy_d_distance = k * (distance - length)

            d_energy_d_x1 = (x1 - x2) * d_energy_d_distance / distance
            d_energy_d_y1 = (y1 - y2) * d_energy_d_distance / distance
            d_energy_d_z1 = (z1 - z2) * d_energy_d_distance / distance

            d_energy_d_x2 = (x2 - x1) * d_energy_d_distance / distance
            d_energy_d_y2 = (y2 - y1) * d_energy_d_distance / distance
            d_energy_d_z2 = (z2 - z1) * d_energy_d_distance / distance

            grad[particle1, 0] -= d_energy_d_x1
            grad[particle1, 1] -= d_energy_d_y1
            grad[particle1, 2] -= d_energy_d_z1

            grad[particle2, 0] -= d_energy_d_x2
            grad[particle2, 1] -= d_energy_d_y2
            grad[particle2, 2] -= d_energy_d_z2
