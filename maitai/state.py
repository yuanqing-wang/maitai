import taichi as ti

def init_state(system):
    state = ti.Struct.field(
        {
            "position": ti.types.vector(3, ti.f32),
            "velocity": ti.types.vector(3, ti.f32),
            "grad": ti.types.vector(3, ti.f32),
            "energy": ti.f32,
        },
        shape=system.n_atoms,
    )

    state.position.fill(0.0)
    state.velocity.fill(0.0)
    state.grad.fill(0.0)
    state.energy.fill(0.0)
    return state
