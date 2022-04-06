import pytest

def test_simulation_construct():
    import taichi as ti
    import numpy as np
    ti.init()

    import maitai as mt
    force = mt.forces.HarmonicBondForce()
    force.add_bond(0, 1, 1.0, 1.0)

    integrator = mt.integrators.DummyIntegrator()
    system = mt.System(forces=[force])
    simulation = mt.Simulation(system=system, integrator=integrator)
