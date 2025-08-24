import sys
from torch import Tensor


def get_size(obj, seen: set = None) -> int:
    """Recursively finds the size of an object in bytes."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)
    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) + get_size(k, seen)
                    for k, v in obj.items()])
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        if isinstance(obj, Tensor):
            size += (obj.element_size()*obj.nelement())
        else:
            size += sum(get_size(i, seen) for i in obj)

    return size


def mem_string(bytes: int) -> str:
    size = bytes
    power = 1024
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    i = 0

    while size >= power and i < len(units) - 1:
        size /= power
        i += 1

    return f"{size:.2f}{units[i]}"
