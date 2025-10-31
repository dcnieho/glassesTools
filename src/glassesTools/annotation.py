from enum import Enum, auto

from . import json, utils

class Type(Enum):
    Point       = auto()
    Interval    = auto()

class EventType(utils.AutoName):
    Validate    = auto()    # interval to be used for running glassesValidator
    Sync_Camera = auto()    # point to be used for synchronizing different cameras
    Sync_ET_Data= auto()    # episode to be used for synchronization of eye tracker data to scene camera (e.g. using VOR)
    Trial       = auto()    # episode for which to map gaze to plane(s): output for files to be provided to user
events = [x.value for x in EventType]
json.register_type(json.TypeEntry(EventType,'__enum.Event__', utils.enum_val_2_str, lambda x: getattr(EventType, x.split('.')[1])))

type_map = {
    EventType.Validate    : Type.Interval,
    EventType.Sync_Camera : Type.Point,
    EventType.Sync_ET_Data: Type.Interval,
    EventType.Trial       : Type.Interval,
}

tooltip_map = {
    EventType.Validate    : 'Validation episode',
    EventType.Sync_Camera : 'Camera sync point',
    EventType.Sync_ET_Data: 'Eye tracker synchronization episode',
    EventType.Trial       : 'Trial episode',
}

def flatten_annotation_dict(annotations: dict[EventType, list[list[int]]]) -> dict[EventType, list[int]]:
    annotations_flat: dict[EventType, list[int]] = {}
    for e in EventType:  # iterate over this for consistent ordering
        if e not in annotations:
            continue
        if annotations[e] and isinstance(annotations[e][0],list):
            annotations_flat[e] = [i for iv in annotations[e] for i in iv]
        else:
            annotations_flat[e] = annotations[e].copy()
    return annotations_flat