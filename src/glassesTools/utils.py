import enum
import pathlib
import typing
import types
import numpy as np
import os
import colorsys


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

def get_colors(n_colors: int, saturation: float, value: float) -> list[tuple[float, float, float]]:
    color_steps = 1/(n_colors+1)
    return [colorsys.hsv_to_rgb(i*color_steps, saturation, value) for i in range(n_colors)]

def get_hour_minutes_seconds_ms(dur_seconds: float) -> tuple[float, float, float, float]:
    hours, remainder = divmod(dur_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    seconds, ms      = divmod(seconds, 1)
    return hours, minutes, seconds, ms
def format_duration(dur: float, show_ms: bool) -> str:
    hours, minutes, seconds, ms = get_hour_minutes_seconds_ms(dur)
    if round(ms,3)==1.:
        # prevent getting timecode x:xx:xx.1000
        hours, minutes, seconds, ms = get_hour_minutes_seconds_ms(round(dur))
    dur_str = f'{int(hours)}:{int(minutes):02d}:{int(seconds):02d}'
    if show_ms:
        dur_str += f'.{ms*1000:03.0f}'
    return dur_str


class AutoName(enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.strip("_").replace("__", "-").replace("_", " ")

def enum_val_2_str(x) -> str:
    # to ensure that string representation of enum is constant over Python versions (it isn't for enum.IntEnum at least)
    return f'{type(x).__name__}.{x.name}'

E = typing.TypeVar('E')
def enum_str_2_val(x: str, enum_cls: E, patches: dict[str,str]=None) -> E:
    str_val = x.split('.')[-1]
    if patches is not None and str_val in patches:
        str_val = patches[str_val]
    return getattr(enum_cls, str_val)


def cartesian_product(*arrays):
    ndim = len(arrays)
    return (np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim))


def fast_scandir(dirname) -> list[pathlib.Path]:
    if not dirname.is_dir():
        return []
    subfolders= [pathlib.Path(f.path) for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

def unpack_none_union(annotation: typing.Type) -> tuple[typing.Type, bool]:
    # below handles both types.Optional and direct unions with None
    if typing.get_origin(annotation) in [typing.Union, types.UnionType] and (args:=typing.get_args(annotation))[-1]==types.NoneType:
        return typing.Union[args[:-1]], True
    else:
        return annotation, False


def set_all(inp: dict[int, bool], value, subset: list[int] = None, predicate: typing.Callable[[int], bool] = None):
    if subset is None:
        subset = (r for r in inp)
    for r in subset:
        if r in inp and (not predicate or predicate(r)):
            inp[r] = value


def trim_str(text: str, length=None, till_newline=True, newline_ellipsis=False):
    if text and till_newline:
        temp = text.splitlines()
        if temp:
            text = temp[0]
        if len(temp)>1 and newline_ellipsis:
            text += '..'
    if length:
        text = (text[:length-2] + '..') if len(text) > length else text
    return text