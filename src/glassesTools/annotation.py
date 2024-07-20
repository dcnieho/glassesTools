from enum import Enum, auto

from . import utils

class Type(Enum):
    Point       = auto()
    Interval    = auto()

class Event(utils.AutoName):
    Validate    = auto()    # interval to be used for running glassesValidator
    Sync_Camera = auto()    # point to be used for synchronizing different cameras
    Sync_ET_Data= auto()    # episode to be used for synchronization of eye tracker data to sceen camera (e.g. using VOR)
    Trial       = auto()    # episode for which to map gaze to plane(s): output for files to be provided to user
events = [x.value for x in Event]
utils.register_type(utils.CustomTypeEntry(Event,'__enum.Event__',str, lambda x: getattr(Event, x.split('.')[1])))

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
