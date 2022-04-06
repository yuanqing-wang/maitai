import taichi as ti

@ti.data_oriented
class System(object):
    def __init__(self, forces=None):
        self.forces = forces
