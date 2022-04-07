import pytest
import numpy.testing as npt
from typing import Tuple

def test_initialize_and_get_energy():
    import taichi as ti
    import numpy as np
    ti.init()

    import maitai as mt
    force = mt.forces.HarmonicBondForce()
    force.add_bond(0, 1, 1.0, 1.0)

    position = ti.field(ti.f32, (2, 3), needs_grad=True)
    position.from_numpy(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        )
    )

    energy = ti.field(ti.f32, (), needs_grad=True)
    energy[None] = 0.0

    @ti.kernel
    def test_energy():
        _energy = force.get_energy(position)
        energy[None] += _energy

    @ti.kernel
    def test_grad():
        force.get_grad(position, grad)

    with ti.Tape(energy):
        test_energy()

    grad = ti.field(ti.f32, (2, 3))
    grad.from_numpy(np.zeros((2, 3)))

    test_grad()

    import math
    npt.assert_almost_equal(
        energy.to_numpy(),
        0.5 * 1.0 * (math.sqrt(3) - 1.0) ** 2
    )

    npt.assert_almost_equal(
        -position.grad.to_numpy(),
        grad.to_numpy(),
    )
