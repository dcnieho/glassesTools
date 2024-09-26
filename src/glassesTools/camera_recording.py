import dataclasses
import pathlib
import typing
import shutil
import json
import os

from . import utils, video_utils


@dataclasses.dataclass
class Recording:
    default_json_file_name      : typing.ClassVar[str] = 'recording_info.json'

    name                        : str
    video_file                  : str
    source_directory            : pathlib.Path
    working_directory           : pathlib.Path  = ""
    duration                    : int           = None

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


    def get_video_path(self):
        vid = self.working_directory / self.video_file
        if not vid.is_file():
            if not self.source_directory.is_absolute():
                vid = (self.working_directory / self.source_directory / self.video_file).resolve()
            else:
                vid = self.source_directory / self.video_file
        return vid



def do_import(rec_info: Recording, cam_cal_file: str|pathlib.Path=None, copy_video=True, source_dir_as_relative_path = False):
    if not rec_info.working_directory:
        raise ValueError('working_directory must be set on the rec_info object')
    rec_info.working_directory = pathlib.Path(rec_info.working_directory)
    ifile = (rec_info.source_directory / rec_info.video_file)
    if not ifile.is_file():
        raise FileNotFoundError(f'The camera recording file {ifile} was not found')
    print(f'processing: {rec_info.video_file} -> {rec_info.working_directory}')

    if not rec_info.working_directory.is_dir():
        rec_info.working_directory.mkdir()

    if copy_video:
        ofile = rec_info.working_directory / rec_info.video_file
        print('  Copy video file...')
        shutil.copy2(ifile, ofile)

    # also get its calibration
    print('  Getting camera calibration...')
    if cam_cal_file is not None:
        shutil.copy2(str(cam_cal_file), str(rec_info.working_directory / 'calibration.xml'))
    else:
        print('  !! No camera calibration provided! Defaulting to hardcoded')

    # and frame timestamps
    print('  Getting frame timestamps...')
    ts = video_utils.get_frame_timestamps_from_video(rec_info.get_video_path())
    ts.to_csv(str(rec_info.working_directory / 'frameTimestamps.tsv'), sep='\t')
    rec_info.duration = ts.timestamp.iat[-1]-ts.timestamp.iat[0]

    # store recording info to folder
    if source_dir_as_relative_path:
        rec_info.source_directory = pathlib.Path(os.path.relpath(rec_info.source_directory,rec_info.working_directory))
    rec_info.store_as_json(rec_info.working_directory)

    return rec_info