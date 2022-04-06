import pytest
import numpy.testing as npt

def test_initialize_and_get_energy():
    import taichi as ti
    import numpy as np
    ti.init()

    import maitai as mt
    force = mt.forces.HarmonicBondForce()
    force.add_bond(0, 1, 1.0, 1.0)

    geometry = ti.field(ti.f32, (2, 3))
    geometry.from_numpy(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        )
    )

    @ti.kernel
    def test_kernel() -> float:
        get_energy = force.get_energy(geometry)
        return get_energy

    get_energy = test_kernel()

    import math
    npt.assert_almost_equal(
        get_energy,
        0.5 * 1.0 * (math.sqrt(3) - 1.0) ** 2
    )
