from enum import Enum, auto

from . import utils

class Type(Enum):
    Point       = auto()
    Interval    = auto()

class Event(utils.AutoName):
    Validate    = auto()    # interval to be used for running glassesValidator
    Sync_Camera = auto()    # point to be used for synchronizing different cameras
    Sync_ET_Data= auto()    # episode to be used for synchronization of eye tracker data to scene camera (e.g. using VOR)
    Trial       = auto()    # episode for which to map gaze to plane(s): output for files to be provided to user
events = [x.value for x in Event]
utils.register_type(utils.CustomTypeEntry(Event,'__enum.Event__', utils.enum_val_2_str, lambda x: getattr(Event, x.split('.')[1])))

type_map = {
    Event.Validate    : Type.Interval,
    Event.Sync_Camera : Type.Point,
    Event.Sync_ET_Data: Type.Interval,
    Event.Trial       : Type.Interval,
}

tooltip_map = {
    Event.Validate    : 'Validation episode',
    Event.Sync_Camera : 'Camera sync point',
    Event.Sync_ET_Data: 'Eye tracker synchronization episode',
    Event.Trial       : 'Trial',
}

def flatten_annotation_dict(annotations: dict[Event, list[list[int]]]) -> dict[Event, list[int]]:
    annotations_flat: dict[Event, list[int]] = {}
    for e in Event:  # iterate over this for consistent ordering
        if e not in annotations:
            continue
        if annotations[e] and isinstance(annotations[e][0],list):
            annotations_flat[e] = [i for iv in annotations[e] for i in iv]
        else:
            annotations_flat[e] = annotations[e].copy()
    return annotations_flat