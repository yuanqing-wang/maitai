def unify_unit(cls):
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        new_args = []
        new_kwargs = {}

        for arg in args:
            if "quantity" in type(args).__name__.lower():
                from . import unit
