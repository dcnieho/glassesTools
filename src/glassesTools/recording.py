import dataclasses
import typing
import pathlib
import json
import natsort

from .eyetracker import EyeTracker
from .timestamps import Timestamp
from . import utils

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
    scene_video_file            : str           = ""


    def store_as_json(self, path: str | pathlib.Path):
        path = pathlib.Path(path)
        if path.is_dir():
            path /= self.default_json_file_name
        with open(path, 'w') as f:
            # remove any crap potentially added by subclasses
            to_dump = dataclasses.asdict(self)
            to_dump = {k:to_dump[k] for k in to_dump if k in Recording.__annotations__ and k not in ['working_directory']}      # working_directory will be loaded as the provided path, and shouldn't be stored
            # dump to file
            json.dump(to_dump, f, cls=utils.CustomTypeEncoder, indent=2)

    @staticmethod
    def load_from_json(path: str | pathlib.Path):
        path = pathlib.Path(path)
        if path.is_dir():
            path /= Recording.default_json_file_name
        with open(path, 'r') as f:
            return Recording(**json.load(f, object_hook=utils.json_reconstitute), working_directory=path.parent)


    def get_scene_video_path(self):
        vid = self.working_directory / self.scene_video_file
        if not vid.is_file():
            if not self.source_directory.is_absolute():
                vid = (self.working_directory / self.source_directory / self.scene_video_file).resolve()
            else:
                vid = self.source_directory / self.scene_video_file
        return vid


def find_recordings(paths: list[pathlib.Path], eye_tracker: EyeTracker):
    from . import importing
    all_recs = []
    for p in paths:
        all_dirs = utils.fast_scandir(p)
        all_dirs.append(p)
        for d in all_dirs:
            # check if dir is a valid recording
            if (recs:=importing.get_recording_info(d, eye_tracker)) is not None:
                all_recs.extend(recs)

    # sort in order natural for OS
    return natsort.os_sorted(all_recs, lambda rec: rec.source_directory)