import pytest

def test_simulation_construct():
    import taichi as ti
    import numpy as np
    ti.init()

    import maitai as mt
    force = mt.forces.HarmonicBondForce()
    force.add_bond(0, 1, 1.0, 1.0)

    integrator = mt.integrators.DummyIntegrator()
    system = mt.System(masses=[1.0, 1.0], forces=[force])
    simulation = mt.Simulation(system=system, integrator=integrator)

    # position = ti.field(ti.f32, (2, 3))
    # position.from_numpy(
    #     np.array(
    #         [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    #     )
    # )

    position = np.array(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    )

    simulation.set_position(position)
    energy = simulation._energy
