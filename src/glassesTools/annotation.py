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

@dataclasses.dataclass
class Event:
    event_type  : EventType
    name        : str
    description : str = ''
    hotkey      : str = ''

EVENT_REGISTRY = []
def register_event(entry: Event):
    EVENT_REGISTRY.append(entry)

def get_event_by_name(name: str) -> Event:
    for e in EVENT_REGISTRY:
        if e.name == name:
            return e
    raise ValueError(f'No event with name {name} is registered')

def get_all_event_names() -> list[str]:
    return [e.name for e in EVENT_REGISTRY]

def get_events_by_type(event_type: EventType) -> list[Event]:
    return [e for e in EVENT_REGISTRY if e.event_type == event_type]

def get_event_type(event_name: str) -> Type:
    event = get_event_by_name(event_name)
    return type_map[event.event_type]


def flatten_annotation_dict(annotations: dict[str, list[list[int]]]) -> dict[str, list[int]]:
    annotations_flat: dict[str, list[int]] = {}
    for e in EVENT_REGISTRY:  # iterate over this for consistent ordering
        if e.name not in annotations:
            continue
        if annotations[e.name] and isinstance(annotations[e.name][0],list):
            annotations_flat[e.name] = [i for iv in annotations[e.name] for i in iv]
        else:
            annotations_flat[e.name] = annotations[e.name].copy()
    return annotations_flat