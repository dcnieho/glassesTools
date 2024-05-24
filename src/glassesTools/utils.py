import enum
import json
import pathlib
import dataclasses
import typing
import numpy as np


def hex_to_rgba_0_1(hex):
    r = int(hex[1:3], base=16) / 255
    g = int(hex[3:5], base=16) / 255
    b = int(hex[5:7], base=16) / 255
    if len(hex) > 7:
        a = int(hex[7:9], base=16) / 255
    else:
        a = 1.0
    return (r, g, b, a)

def rgba_0_1_to_hex(rgba):
    r = "%.2x" % int(rgba[0] * 255)
    g = "%.2x" % int(rgba[1] * 255)
    b = "%.2x" % int(rgba[2] * 255)
    if len(rgba) > 3:
        a = "%.2x" % int(rgba[3] * 255)
    else:
        a = "FF"
    return f"#{r}{g}{b}{a}"


class AutoName(enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.strip("_").replace("__", "-").replace("_", " ")


@dataclasses.dataclass
class CustomTypeEntry:
    type        : typing.Type
    reg_name    : str
    to_json     : typing.Callable
    from_json   : typing.Callable

CUSTOM_TYPE_REGISTRY = []
def register_type(entry: CustomTypeEntry):
    CUSTOM_TYPE_REGISTRY.append(entry)
register_type(CustomTypeEntry(pathlib.Path,'pathlib.Path',str,lambda x: pathlib.Path(x)))

class CustomTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        for t in CUSTOM_TYPE_REGISTRY:
            if isinstance(obj, t.type):
                return {t.reg_name: t.to_json(obj)}
        return json.JSONEncoder.default(self, obj)

def json_reconstitute(d):
    for t in CUSTOM_TYPE_REGISTRY:
        if t.reg_name in d:
            return t.from_json(d[t.reg_name])
    return d


def cartesian_product(*arrays):
    ndim = len(arrays)
    return (np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim))
