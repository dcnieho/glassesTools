import orjson
import json
import typing
import dataclasses
import pathlib



def json_encoder(obj):
    matches: dict[typing.Type, TypeEntry] = {}
    for t in TYPE_REGISTRY:
        if isinstance(obj, t.type):
            matches[t.type] = t
    if matches:
        # find most precise match and use that
        t = None
        for t2 in matches:
            if t is None or issubclass(t2,t):
                t = t2
        return {matches[t].reg_name: matches[t].to_json(obj)}
    else:
        raise TypeError(f'type {type(obj)} cannot be serialized')

def dump(obj, file: pathlib.Path):
    data = orjson.dumps(obj, json_encoder, orjson.OPT_INDENT_2|orjson.OPT_PASSTHROUGH_SUBCLASS)
    with open(file, 'wb') as f:
        f.write(data)


def json_decoder(d):
    for t in TYPE_REGISTRY:
        if t.reg_name in d:
            return t.from_json(d[t.reg_name])
        elif t.compatible_reg_names is not None:
            for r_name in t.compatible_reg_names:
                if r_name in d:
                    return t.from_json(d[r_name])
    return d

def load(file: pathlib.Path):
    with open(file, 'r') as f:
        return json.load(f, object_hook=json_decoder)

def loads(payload: str):
    return json.loads(payload, object_hook=json_decoder)


@dataclasses.dataclass
class TypeEntry:
    type                : typing.Type
    reg_name            : str
    to_json             : typing.Callable
    from_json           : typing.Callable
    compatible_reg_names: list[str]|None = None

TYPE_REGISTRY = []
def register_type(entry: TypeEntry):
    TYPE_REGISTRY[:] = [existing for existing in TYPE_REGISTRY if existing.reg_name != entry.reg_name]
    TYPE_REGISTRY.append(entry)

def _normalize_path_separators(path: str):
    if path.startswith('\\\\'):
        return path
    return path.replace('\\','/')

def _path_to_json(path: pathlib.Path):
    return _normalize_path_separators(str(path))

def _path_from_json(path: str):
    return pathlib.Path(_normalize_path_separators(path))

register_type(TypeEntry(pathlib.Path,'pathlib.Path',_path_to_json,_path_from_json))
register_type(TypeEntry(set,'builtin.set',list,lambda x: set(x)))
register_type(TypeEntry(tuple,'builtin.tuple',list,lambda x: tuple(x)))
