import dataclasses
import typing
import pathlib
import json

from .eyetracker import EyeTracker
from .timestamps import Timestamp
from. import utils

@dataclasses.dataclass
class Recording:
    default_json_file_name      : typing.ClassVar[str] = 'recording_info.json'

    name                        : str           = ""
    source_directory            : pathlib.Path  = ""
    working_directory           : pathlib.Path  = ""
    start_time                  : Timestamp     = 0
    duration                    : int           = None
    eye_tracker                 : EyeTracker    = EyeTracker.Unknown
    project                     : str           = ""
    participant                 : str           = ""
    firmware_version            : str           = ""
    glasses_serial              : str           = ""
    recording_unit_serial       : str           = ""
    recording_software_version  : str           = ""
    scene_camera_serial         : str           = ""

    def store_as_json(self, path: str | pathlib.Path):
        path = pathlib.Path(path)
        if path.is_dir():
            path /= self.default_json_file_name
        with open(path, 'w') as f:
            # remove any crap potentially added by subclasses
            to_dump = dataclasses.asdict(self)
            to_dump = {k:to_dump[k] for k in to_dump if k in Recording.__annotations__}
            # dump to file
            json.dump(to_dump, f, cls=utils.CustomTypeEncoder)

    @classmethod
    def load_from_json(cls, path: str | pathlib.Path):
        path = pathlib.Path(path)
        if path.is_dir():
            path /= cls.default_json_file_name
        with open(path, 'r') as f:
            return cls(**json.load(f, object_hook=utils.json_reconstitute))
