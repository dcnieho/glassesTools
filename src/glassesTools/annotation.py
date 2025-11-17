from enum import Enum, auto
import dataclasses

from . import json, utils

class Type(Enum):
    Point       = auto()
    Interval    = auto()

class EventType(utils.AutoName):
    Validate    = auto()    # interval to be used for running glassesValidator
    Sync_Camera = auto()    # point to be used for synchronizing different cameras
    Sync_ET_Data= auto()    # episode to be used for synchronization of eye tracker data to scene camera (e.g. using VOR)
    Trial       = auto()    # episode for which to map gaze to plane(s): output for files to be provided to user
event_types = [x for x in EventType]
json.register_type(json.TypeEntry(EventType,'__enum.Event__', utils.enum_val_2_str, lambda x: getattr(EventType, x.split('.')[1])))

type_map = {
    EventType.Validate      : Type.Interval,
    EventType.Sync_Camera   : Type.Point,
    EventType.Sync_ET_Data  : Type.Interval,
    EventType.Trial         : Type.Interval,
}

tooltip_map = {
    EventType.Validate      : 'Validation episode',
    EventType.Sync_Camera   : 'Camera sync point',
    EventType.Sync_ET_Data  : 'Eye tracker synchronization episode',
    EventType.Trial         : 'Trial episode',
}

default_hotkeys = {
    EventType.Validate      : 'v',
    EventType.Sync_Camera   : 'c',
    EventType.Sync_ET_Data  : 'e',
    EventType.Trial         : 't',
}

@dataclasses.dataclass
class Event:
    event_type  : EventType
    name        : str
    description : str = ''
    hotkey      : str = ''

EVENT_REGISTRY = []
def register_event(entry: Event):
    EVENT_REGISTRY.append(entry)

def unregister_all_annotation_types():
    EVENT_REGISTRY.clear()

def get_events_by_type(event_type: EventType) -> list[Event]:
    return [e for e in EVENT_REGISTRY if e.event_type == event_type]


def flatten_annotation_dict(annotations: dict[str, tuple[EventType, list[list[int]]]]) -> dict[str, tuple[EventType, list[int]]]:
    annotations_flat: dict[str, tuple[EventType, list[int]]] = {}
    def _copy_flat_annotation(annotations: tuple[EventType, list[list[int]]]):
        if annotations[1] and isinstance(annotations[1][0],list):
            return (annotations[0], [i for iv in annotations[1] for i in iv])
        else:
            return (annotations[0], annotations[1].copy())
    for e in EVENT_REGISTRY:  # iterate over this for consistent ordering
        if e.name not in annotations:
            continue
        annotations_flat[e.name] = _copy_flat_annotation(annotations[e.name])
    # add anything still missing
    for e_name in annotations:
        if e_name not in annotations_flat:
            annotations_flat[e_name] = _copy_flat_annotation(annotations[e_name])
    return annotations_flat