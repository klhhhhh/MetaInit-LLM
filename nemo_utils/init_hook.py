from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

INIT_RECORDS = {}

def patch_layer(cls, cls_name):
    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        # Call the original constructor
        original_init(self, *args, **kwargs)

        # Record constructor arguments (including class object, class name, args, kwargs)
        INIT_RECORDS[id(self)] = {
            "class_name": cls_name,
            "instance": self,
            "args": args,
            "kwargs": kwargs
        }

    cls.__init__ = patched_init