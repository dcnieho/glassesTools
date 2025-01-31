"""
Copy data already in common format into a glassesTools recording.
Name of recording will be the name of the folder that is imported.
"""

import shutil
import pathlib
import pandas as pd

from ..recording import Recording
from ..eyetracker import EyeTracker
from .. import gaze_headref
from .. import naming, video_utils

def importData(output_dir: str|pathlib.Path=None, source_dir: str|pathlib.Path=None, rec_info: Recording=None, device_name: str=None, copy_scene_video = True, source_dir_as_relative_path = False, cam_cal_file: str|pathlib.Path=None) -> Recording:
    from . import check_folders, _store_data
    output_dir, source_dir, rec_info, device_name = check_folders(output_dir, source_dir, rec_info, EyeTracker.Generic, device_name)
    print(f'processing: {source_dir.name} -> {output_dir}')

    ### check and copy needed files to the output directory
    print('  Check and copy raw data...')
    ### check recording and get export directory
    if rec_info is not None:
        checkRecording(source_dir, rec_info, device_name)
    else:
        rec_info = getRecordingInfo(source_dir, device_name)
        if rec_info is None:
            raise RuntimeError(f"The folder {source_dir} is not recognized as a {EyeTracker.Generic.value} recording.")

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()


    ### copy the raw data to the output directory
    srcVid, destVid, gotFrameTs, gotCal = copyGenericRecording(source_dir, output_dir, copy_scene_video, cam_cal_file)
    if destVid:
        rec_info.scene_video_file = destVid.name
    else:
        rec_info.scene_video_file =  srcVid.name

    if not gotFrameTs:
        frameTimestamps = video_utils.get_frame_timestamps_from_video(rec_info.get_scene_video_path())
    else:
        frameTimestamps = None

    if not gotCal:
        print('    !! No camera calibration provided!')

    if not rec_info.duration:
        # if duration not known, fill it
        # make a reasonable estimate of duration
        gaze = gaze_headref.read_dict_from_file(output_dir/naming.gaze_data_fname)[0]
        framets = frameTimestamps   # local copy to not accidentally trigger any overwriting in _store_data() below
        if framets is None:
            framets = pd.read_csv(output_dir/naming.frame_timestamps_fname, delimiter='\t', index_col='frame_idx')
        gt0 = gaze[min(gaze.keys())][0].timestamp_ori
        gte = gaze[max(gaze.keys())][-1].timestamp_ori
        rec_info.duration = round(max(gte-gt0,framets.timestamp.iat[-1]))

    _store_data(output_dir, None, frameTimestamps, rec_info, source_dir_as_relative_path=source_dir_as_relative_path)

    return rec_info

def getRecordingInfo(inputDir: str|pathlib.Path, device_name: str=None) -> Recording:
    # returns None if not a recording directory
    inputDir = pathlib.Path(inputDir)

    rec_info_fname = inputDir/Recording.default_json_file_name
    if rec_info_fname.is_file():
        recInfo = Recording.load_from_json(rec_info_fname)
        if recInfo.eye_tracker!=EyeTracker.Generic:
            print(f"A recording for a \"{EyeTracker.Generic.value}\" eye tracker was not found in the folder {inputDir}.")
            return None
        if recInfo.eye_tracker_name!=device_name:
            print(f"A recording for a \"{device_name}\" device was not found in the folder {inputDir}.")
            return None
        # override inputDir to make sure its set correctly
        recInfo.source_directory = inputDir
    else:
        recInfo = Recording(source_directory=inputDir, eye_tracker=EyeTracker.Generic, eye_tracker_name=device_name)
        # get recording info
        recInfo.name = inputDir.name

    # check expected files are present
    for f in ('worldCamera.mp4','gazeData.tsv'):
        if not (inputDir/f).is_file():
            print(f'This directory does not contain a valid generic recording for a {device_name} eye tracker. The {f} file is not found in the input directory {inputDir}.')
            return None

    # we got a valid recording
    # return what we've got
    return recInfo

def checkRecording(inputDir: str|pathlib.Path, recInfo: Recording, device_name: str=None):
    actualRecInfo = getRecordingInfo(inputDir, device_name)

    if actualRecInfo is None or recInfo.name!=actualRecInfo.name:
        raise ValueError(f"A recording with the name \"{recInfo.name}\" was not found in the folder {inputDir}.")

    # make sure caller did not mess with recInfo
    if recInfo.eye_tracker_name!=actualRecInfo.eye_tracker_name:
        raise ValueError(f"A recording for a \"{recInfo.eye_tracker_name}\" device was not found in the folder {inputDir}.")

def copyGenericRecording(inputDir: pathlib.Path, outputDir: pathlib.Path, copy_scene_video:bool, cam_cal_file: str|pathlib.Path|None):
    gazeFile = inputDir/'gazeData.tsv'
    if not gazeFile.is_file():
        raise RuntimeError(f'The {gazeFile} file is not found in the input directory {inputDir}')
    shutil.copy2(gazeFile, outputDir / naming.gaze_data_fname)

    vidSrcFile = inputDir/'worldCamera.mp4'
    if not vidSrcFile.is_file():
        raise RuntimeError(f'The {vidSrcFile} file is not found in the input directory {inputDir}')
    if copy_scene_video:
        vidDestFile = outputDir / f'{naming.scene_camera_video_fname_stem}.mp4'
        shutil.copy2(vidSrcFile, vidDestFile)
    else:
        vidDestFile = None

    frame_ts_file = inputDir/'frameTimestamps.tsv'
    gotFrameTs = frame_ts_file.is_file()
    if gotFrameTs:
        shutil.copy2(frame_ts_file, outputDir / naming.frame_timestamps_fname)

    if cam_cal_file is not None:
        shutil.copy2(cam_cal_file, outputDir / naming.scene_camera_calibration_fname)
        gotCal = True
    else:
        cal_file = inputDir/'calibration.xml'
        gotCal = cal_file.is_file()
        if gotCal:
            shutil.copy2(cal_file, outputDir / naming.scene_camera_calibration_fname)

    return vidSrcFile, vidDestFile, gotFrameTs, gotCal