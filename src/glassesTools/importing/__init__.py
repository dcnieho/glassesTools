# -*- coding: utf-8 -*-
import pathlib
import os
import pandas as pd
import polars as pl

from ..recording import Recording
from ..eyetracker import EyeTracker
from .. import eyetracker, naming

from .adhawk_mindlink import preprocessData as adhawk_mindlink
from .generic import importData as generic
from .meta_aria_gen1 import importData as meta_aria_gen1
from .SeeTrue_STONE import preprocessData as SeeTrue_STONE
from .SMI_ETG import preprocessData as SMI_ETG
from .tobii_G2 import preprocessData as tobii_G2
from .tobii_G3 import preprocessData as tobii_G3
from .VPS_19 import preprocessData as VPS_19

def pupil_core(output_dir: str | pathlib.Path, source_dir: str | pathlib.Path = None, rec_info: Recording = None, copy_scene_video = True, source_dir_as_relative_path = False) -> Recording:
    from .pupilLabs import preprocessData
    return preprocessData(output_dir, EyeTracker.Pupil_Core, source_dir, rec_info, copy_scene_video, source_dir_as_relative_path)

def pupil_invisible(output_dir: str | pathlib.Path, source_dir: str | pathlib.Path = None, rec_info: Recording = None, copy_scene_video = True, source_dir_as_relative_path = False) -> Recording:
    from .pupilLabs import preprocessData
    return preprocessData(output_dir, EyeTracker.Pupil_Invisible, source_dir, rec_info, copy_scene_video, source_dir_as_relative_path)

def pupil_neon(output_dir: str | pathlib.Path, source_dir: str | pathlib.Path = None, rec_info: Recording = None, copy_scene_video = True, source_dir_as_relative_path = False) -> Recording:
    from .pupilLabs import preprocessData
    return preprocessData(output_dir, EyeTracker.Pupil_Neon, source_dir, rec_info, copy_scene_video, source_dir_as_relative_path)


def get_recording_info(source_dir: str | pathlib.Path, device: str | EyeTracker, device_name: str = None) -> list[Recording]:
    source_dir  = pathlib.Path(source_dir)
    device = eyetracker.string_to_enum(device)

    rec_info = None
    match device:
        case EyeTracker.AdHawk_MindLink:
            from .adhawk_mindlink import getRecordingInfo
            rec_info = getRecordingInfo(source_dir)
        case EyeTracker.Generic:
            from .generic import getRecordingInfo
            rec_info = getRecordingInfo(source_dir, device_name)
        case EyeTracker.Meta_Aria_Gen_1:
            from .meta_aria_gen1 import getRecordingInfo
            rec_info = getRecordingInfo(source_dir)
        case EyeTracker.Pupil_Core:
            from .pupilLabs import getRecordingInfo
            rec_info = getRecordingInfo(source_dir, device)
        case EyeTracker.Pupil_Invisible:
            from .pupilLabs import getRecordingInfo
            rec_info = getRecordingInfo(source_dir, device)
        case EyeTracker.Pupil_Neon:
            from .pupilLabs import getRecordingInfo
            rec_info = getRecordingInfo(source_dir, device)
        case EyeTracker.SeeTrue_STONE:
            from .SeeTrue_STONE import getRecordingInfo
            rec_info = getRecordingInfo(source_dir)
        case EyeTracker.SMI_ETG:
            from .SMI_ETG import getRecordingInfo
            rec_info = getRecordingInfo(source_dir)
        case EyeTracker.Tobii_Glasses_2:
            from .tobii_G2 import getRecordingInfo
            rec_info = getRecordingInfo(source_dir)
        case EyeTracker.Tobii_Glasses_3:
            from .tobii_G3 import getRecordingInfo
            rec_info = getRecordingInfo(source_dir)
        case EyeTracker.VPS_19:
            from .VPS_19 import getRecordingInfo
            rec_info = getRecordingInfo(source_dir)

    if rec_info is not None and not isinstance(rec_info,list):
        rec_info = [rec_info]
    return rec_info


# single front end to the various device import functions for convenience
def do_import(output_dir: str | pathlib.Path = None, source_dir: str | pathlib.Path = None, device: str | EyeTracker = None, rec_info: Recording = None, copy_scene_video = True, source_dir_as_relative_path = False, cam_cal_file: str|pathlib.Path=None, device_name: str=None) -> Recording:
    # output_dir is the working directory folder where the export of this recording will be placed
    # should match rec_info.working_directory if both are provided (is checked below)
    # NB: cam_cal_file input is only honored for AdHawk MindLink and SeeTrue STONE recordings
    if rec_info is not None:
        if isinstance(rec_info,list):
            raise ValueError('You should provide a single Recording to this function''s "rec_info" input argument, not a list of Recordings.')
    device    , rec_info, device_name = check_device(device, rec_info, device_name)
    source_dir, rec_info = check_source_dir(source_dir, rec_info)
    output_dir, rec_info = check_output_dir(output_dir, rec_info)

    # do the actual import/pre-process
    match device:
        case EyeTracker.AdHawk_MindLink:
            rec_info = adhawk_mindlink(output_dir, source_dir, rec_info, copy_scene_video=copy_scene_video, source_dir_as_relative_path=source_dir_as_relative_path, cam_cal_file=cam_cal_file)
        case EyeTracker.Generic:
            rec_info = generic(output_dir, source_dir, rec_info, device_name, copy_scene_video=copy_scene_video, source_dir_as_relative_path=source_dir_as_relative_path, cam_cal_file=cam_cal_file)
        case EyeTracker.Meta_Aria_Gen_1:
            rec_info = meta_aria_gen1(output_dir, source_dir, rec_info, copy_scene_video=copy_scene_video, source_dir_as_relative_path=source_dir_as_relative_path)
        case EyeTracker.Pupil_Core:
            rec_info = pupil_core(output_dir, source_dir, rec_info, copy_scene_video=copy_scene_video, source_dir_as_relative_path=source_dir_as_relative_path)
        case EyeTracker.Pupil_Invisible:
            rec_info = pupil_invisible(output_dir, source_dir, rec_info, copy_scene_video=copy_scene_video, source_dir_as_relative_path=source_dir_as_relative_path)
        case EyeTracker.Pupil_Neon:
            rec_info = pupil_neon(output_dir, source_dir, rec_info, copy_scene_video=copy_scene_video, source_dir_as_relative_path=source_dir_as_relative_path)
        case EyeTracker.SeeTrue_STONE:
            rec_info = SeeTrue_STONE(output_dir, source_dir, rec_info, copy_scene_video=copy_scene_video, source_dir_as_relative_path=source_dir_as_relative_path, cam_cal_file=cam_cal_file)
        case EyeTracker.SMI_ETG:
            rec_info = SMI_ETG(output_dir, source_dir, rec_info, copy_scene_video=copy_scene_video, source_dir_as_relative_path=source_dir_as_relative_path)
        case EyeTracker.Tobii_Glasses_2:
            rec_info = tobii_G2(output_dir, source_dir, rec_info, copy_scene_video=copy_scene_video, source_dir_as_relative_path=source_dir_as_relative_path)
        case EyeTracker.Tobii_Glasses_3:
            rec_info = tobii_G3(output_dir, source_dir, rec_info, copy_scene_video=copy_scene_video, source_dir_as_relative_path=source_dir_as_relative_path)
        case EyeTracker.VPS_19:
            rec_info = VPS_19(output_dir, source_dir, rec_info, copy_scene_video=copy_scene_video, source_dir_as_relative_path=source_dir_as_relative_path)

    return rec_info


def check_source_dir(source_dir: str|pathlib.Path, rec_info: Recording) -> tuple[pathlib.Path, Recording]:
    if source_dir is not None:
        source_dir  = pathlib.Path(source_dir)
        if rec_info is not None and rec_info.source_directory and pathlib.Path(rec_info.source_directory) != source_dir:
            raise ValueError(f"The provided source_dir ({source_dir}) does not equal the source directory set in rec_info ({rec_info.source_directory}).")
    elif rec_info is None:
        raise RuntimeError('Either the "input_dir" or the "rec_info" input argument should be set.')
    else:
        source_dir  = pathlib.Path(rec_info.source_directory)

    if rec_info is not None and not rec_info.source_directory:
        rec_info.source_directory = source_dir
    return source_dir, rec_info

def check_output_dir(output_dir: str|pathlib.Path, rec_info: Recording) -> tuple[pathlib.Path, Recording]:
    if output_dir is not None:
        output_dir  = pathlib.Path(output_dir)
        if rec_info is not None and rec_info.working_directory and pathlib.Path(rec_info.working_directory) != output_dir:
            raise ValueError(f"The provided output_dir ({output_dir}) does not equal the working directory set in rec_info ({rec_info.working_directory}).")
    elif rec_info is None:
        raise RuntimeError('Either the "output_dir" or the "rec_info" input argument should be set.')
    else:
        output_dir  = pathlib.Path(rec_info.working_directory)

    if output_dir.is_dir():
        with os.scandir(output_dir) as it:
            if any(it):
                raise RuntimeError(f'Output directory ({output_dir}) already exists and is not empty. Cannot use.')

    if rec_info is not None and not rec_info.working_directory:
        rec_info.working_directory = output_dir
    return output_dir, rec_info

def check_folders(output_dir: str|pathlib.Path, source_dir: str|pathlib.Path, rec_info: Recording, device: EyeTracker, device_name: str=None) -> tuple[pathlib.Path, pathlib.Path, Recording]:
    if rec_info is not None and rec_info.eye_tracker:
        if rec_info.eye_tracker!=device:
            raise ValueError(f'Provided rec_info is for a device ({rec_info.eye_tracker.value}) that is not a {device.value}. Cannot use.')
        if device_name is not None and rec_info.eye_tracker_name!=device_name:
            raise ValueError(f'Provided rec_info is for a {rec_info.eye_tracker.value} device with the name "{rec_info.eye_tracker_name}" that is not the expected value ({device_name}). Cannot use.')
        if not device_name:
            device_name = rec_info.eye_tracker_name

    source_dir, rec_info = check_source_dir(source_dir, rec_info)
    output_dir, rec_info = check_output_dir(output_dir, rec_info)
    return output_dir, source_dir, rec_info, device_name

def check_device(device: str|EyeTracker, rec_info: Recording, device_name: str=None):
    if device is None and (rec_info is None or not rec_info.eye_tracker):
        raise RuntimeError('Either the "device" or the eye_tracker field of the "rec_info" input argument should be set.')
    if device is not None:
        device = eyetracker.string_to_enum(device)
    if rec_info is not None and rec_info.eye_tracker:
        if device is not None:
            if rec_info.eye_tracker != device:
                raise ValueError(f'Provided device ({device.value}) did not match device specified in rec_info ({rec_info.eye_tracker.value}). Provide matching values or do not provide the device input argument.')
        else:
            device = eyetracker.string_to_enum(rec_info.eye_tracker)

    if device_name is not None:
        if device!=EyeTracker.Generic:
            raise RuntimeError(f'The device_name parameter should not be set for devices other than a {EyeTracker.Generic.value} device, but it was set.')
        elif rec_info is not None and device_name!=rec_info.eye_tracker_name:
            raise RuntimeError(f'Provided device_name ({device_name}) did not match the device name specified in rec_info ({rec_info.eye_tracker_name}). Provide matching values or do not provide the device_name input argument.')
    elif device==EyeTracker.Generic:
        if rec_info is not None:
            device_name = rec_info.eye_tracker_name
        if device==EyeTracker.Generic and not device_name:
            raise RuntimeError(f'For a {rec_info.eye_tracker.value} device, the device_name parameter should be set or the eye tracker name should be set in recording info, but it was not')

    return device, rec_info, device_name

def _store_data(output_dir: pathlib.Path, gaze: pd.DataFrame|None, frame_ts: pd.DataFrame|None, rec_info: Recording, gaze_fname = naming.gaze_data_fname, frame_ts_fname = naming.frame_timestamps_fname, rec_info_fname = Recording.default_json_file_name, source_dir_as_relative_path = False):
    # write the gaze data to a csv file (polars as that library saves to file waaay faster)
    if gaze is not None:
        pl.from_pandas(gaze,include_index=True).write_csv(output_dir / gaze_fname, separator='\t', null_value='nan', float_precision=8)

    # also store frame timestamps
    if frame_ts is not None:
        pl.from_pandas(frame_ts,include_index=True).write_csv(output_dir / frame_ts_fname, separator='\t', float_precision=8)

    # store rec info
    if source_dir_as_relative_path:
        rec_info.source_directory = pathlib.Path(os.path.relpath(rec_info.source_directory,output_dir))
    # if duration not known, fill it
    if not rec_info.duration and gaze is not None:
        # make a reasonable estimate of duration
        rec_info.duration = round(max(gaze.index[-1]-gaze.index[0],frame_ts.timestamp.iat[-1]))
    rec_info.store_as_json(output_dir / rec_info_fname)


__all__ = ['adhawk_mindlink','generic','pupil_core','pupil_invisible','pupil_neon','SeeTrue_STONE','SMI_ETG','tobii_G2','tobii_G3','VPS_19',
           'get_recording_info','do_import','check_source_dir','check_output_dir','check_folders','check_device']