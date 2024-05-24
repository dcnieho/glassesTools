# -*- coding: utf-8 -*-
import pathlib
import os

from ..recording import Recording
from ..eyetracker import EyeTracker
from .. import eyetracker

from .adhawk_mindlink import preprocessData as adhawk_mindlink
from .SeeTrue_STONE import preprocessData as SeeTrue_STONE
from .SMI_ETG import preprocessData as SMI_ETG
from .tobii_G2 import preprocessData as tobii_G2
from .tobii_G3 import preprocessData as tobii_G3

def pupil_core(output_dir: str | pathlib.Path, source_dir: str | pathlib.Path = None, rec_info: Recording = None):
    from .pupilLabs import preprocessData
    preprocessData(output_dir, 'Pupil Core', source_dir, rec_info)

def pupil_invisible(output_dir: str | pathlib.Path, source_dir: str | pathlib.Path = None, rec_info: Recording = None):
    from .pupilLabs import preprocessData
    preprocessData(output_dir, 'Pupil Invisible', source_dir, rec_info)

def pupil_neon(output_dir: str | pathlib.Path, source_dir: str | pathlib.Path = None, rec_info: Recording = None):
    from .pupilLabs import preprocessData
    preprocessData(output_dir, 'Pupil Neon', source_dir, rec_info)


def get_recording_info(source_dir: str | pathlib.Path, device: str | EyeTracker):
    source_dir  = pathlib.Path(source_dir)
    device = eyetracker.string_to_enum(device)

    rec_info = None
    match device:
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
        case EyeTracker.AdHawk_MindLink:
            from .adhawk_mindlink import getRecordingInfo
            rec_info = getRecordingInfo(source_dir)

    if rec_info is not None and not isinstance(rec_info,list):
        rec_info = [rec_info]
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

def check_rec_info(rec_info: Recording, eye_tracker: EyeTracker):
    if rec_info is not None:
        if rec_info.eye_tracker!=eye_tracker:
            raise ValueError(f'Provided rec_info is for a device ({rec_info.eye_tracker.value}) that is not a {eye_tracker.value}. Cannot use.')

def check_folders(output_dir: str|pathlib.Path, source_dir: str|pathlib.Path, rec_info: Recording, eye_tracker: EyeTracker) -> tuple[pathlib.Path, pathlib.Path, Recording]:
    check_rec_info(rec_info, eye_tracker)
    source_dir, rec_info = check_source_dir(source_dir, rec_info)
    output_dir, rec_info = check_output_dir(output_dir, rec_info)
    return output_dir, source_dir, rec_info

def check_device(device: str|EyeTracker, rec_info: Recording):
    if device is None and (rec_info is None or not rec_info.eye_tracker):
        raise RuntimeError('Either the "device" or the eye_tracker field of the "rec_info" input argument should be set.')
    if device is not None:
        device = eyetracker.string_to_enum(device)
    if rec_info is not None:
        if device is not None:
            if rec_info.eye_tracker != device:
                raise ValueError(f'Provided device ({device.value}) did not match device specific in rec_info ({rec_info.eye_tracker.value}). Provide matching values or do not provide the device input argument.')
        else:
            device = eyetracker.string_to_enum(rec_info.eye_tracker)
    return device, rec_info


__all__ = ['pupil_core','pupil_invisible','pupil_neon','SeeTrue_STONE','SMI_ETG','tobii_G2','tobii_G3','adhawk_mindlink',
           'get_recording_info']