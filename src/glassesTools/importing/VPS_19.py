"""
Cast raw Viewpointsystem VPS 19 data into common format.

The output directory will contain:
    - frameTimestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gazeData.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera,
                    and 3D coordinates in the coordinate frame of the glasses (may need rotation/translation
                    to represent in world camera's coordinate frame)
"""

import shutil
import pathlib
import json
import pandas as pd
import os
import datetime

from ..recording import Recording
from ..eyetracker import EyeTracker
from .. import naming, timestamps, video_utils


def preprocessData(output_dir: str|pathlib.Path=None, source_dir: str|pathlib.Path=None, rec_info: Recording=None, copy_scene_video = True, source_dir_as_relative_path = False, cam_cal_file: str|pathlib.Path=None) -> Recording:
    from . import check_folders, _store_data
    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, EyeTracker.VPS_19)
    print(f'processing: {source_dir.name} -> {output_dir}')


    ### check and copy needed files to the output directory
    print('  Check and copy raw data...')
    ### check tobii recording and get export directory
    if rec_info is not None:
        checkRecording(source_dir, rec_info)
    else:
        rec_info = getRecordingInfo(source_dir)
        if rec_info is None:
            raise RuntimeError(f"The folder {source_dir} is not recognized as a {EyeTracker.VPS_19.value} recording.")

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()


    ### copy the raw data to the output directory
    copyVPS19Recording(source_dir, output_dir, rec_info, copy_scene_video)

    #### prep the copied data...
    print('  Getting camera calibration...')
    if cam_cal_file is not None:
        shutil.copyfile(cam_cal_file, output_dir / naming.scene_camera_calibration_fname)
    else:
        print('    !! No camera calibration provided!')
    print('  Prepping gaze data...')
    gazeDf, frameTimestamps = formatGazeData(source_dir, rec_info)

    _store_data(output_dir, gazeDf, frameTimestamps, rec_info, source_dir_as_relative_path=source_dir_as_relative_path)

    return rec_info


def getRecordingInfo(inputDir: str|pathlib.Path) -> Recording:
    # get recordings. A folder can contain multiple recordings.
    inputDir = pathlib.Path(inputDir)
    # recordings are identified as a tsv and an mkv file with the
    # same name
    recInfos: list[Recording] = []
    for r in inputDir.glob('*.tsv'):
        if not r.with_suffix('.mkv').is_file():
            continue
        recInfos.append(Recording(source_directory=inputDir, eye_tracker=EyeTracker.VPS_19))
        recInfos[-1].name = r.stem
        # get more info
        with open(r,'rt') as f:
            lines = []
            for _ in range(5):
                lines.append(f.readline())
        time_string = lines[2][len('# Recording start: '):].strip()
        recInfos[-1].start_time = timestamps.Timestamp(int(datetime.datetime.fromisoformat(time_string).timestamp()))
        rInfo = json.loads(lines[4][len('# system: '):-1])
        recInfos[-1].glasses_serial = rInfo['glasses']
        recInfos[-1].recording_unit_serial = rInfo['Smart Unit']
        recInfos[-1].firmware_version = rInfo['operating system']

    # should return None if no valid recordings found
    return recInfos if recInfos else None


def checkRecording(inputDir: str|pathlib.Path, recInfo: Recording):
    # check we have an exported gaze data file
    for ext in ('tsv','mkv'):
        file = f'{recInfo.name}.{ext}'
        if not (inputDir / file).is_file():
            if use_return:
                return False
            else:
                raise RuntimeError(f'Recording {recInfo.name} not found: {file} file not found in {inputDir}.')

    return True

def copyVPS19Recording(inputDir: pathlib.Path, outputDir: pathlib.Path, recInfo: Recording, copy_scene_video: bool):
    """
    Copy the relevant files from the specified input dir to the specified output dir
    """
    # Copy relevant files to new directory
    srcFile  = inputDir / f'{recInfo.name}.mkv'

    if copy_scene_video:
        destFile = outputDir / f'{naming.scene_camera_video_fname_stem}.mkv'
        shutil.copy2(srcFile, destFile)
    else:
        destFile = None

    if destFile:
        recInfo.scene_video_file = destFile.name
    else:
        recInfo.scene_video_file =  srcFile.name


def formatGazeData(inputDir: str|pathlib.Path, recInfo: Recording):
    """
    load gazedata json file
    format to get the gaze coordinates w/r/t world camera, and timestamps for every frame of video

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps
    """

    # convert the json file to pandas dataframe
    df = gaze2df(inputDir / f'{recInfo.name}.tsv')

    # read video file, create array of frame timestamps
    frameTimestamps = video_utils.get_frame_timestamps_from_video(recInfo.get_scene_video_path())

    # return the gaze data df and frame time stamps array
    return df, frameTimestamps


def gaze2df(gazeFile: str|pathlib.Path) -> pd.DataFrame:
    """
    convert the .tsv file to a pandas dataframe
    """

    df = pd.read_csv(gazeFile,delimiter='\t',comment='#',index_col=False)

    # get time sync info
    t0 = df['FrontTimeStamp'].iat[0]-df['MediaTimeStamp'].iat[0]

    # rename and reorder columns
    lookup = {'GazeTimeStamp': 'timestamp',
              'MediaFrameIndex': 'frame_idx',
               'Gaze2dX':'gaze_pos_vid_x',
               'Gaze2dY':'gaze_pos_vid_y',
               'PupilCenterLeftX':'gaze_ori_l_x',
               'PupilCenterLeftY':'gaze_ori_l_y',
               'PupilCenterLeftZ':'gaze_ori_l_z',
               'GazeLeftX':'gaze_dir_l_x',
               'GazeLeftY':'gaze_dir_l_y',
               'GazeLeftZ':'gaze_dir_l_z',
               'PupilCenterRightX':'gaze_ori_r_x',
               'PupilCenterRightY':'gaze_ori_r_y',
               'PupilCenterRightZ':'gaze_ori_r_z',
               'GazeRightX':'gaze_dir_r_x',
               'GazeRightY':'gaze_dir_r_y',
               'GazeRightZ':'gaze_dir_r_z',
               'PupilDiaLeft':'pup_diam_l',
               'PupilDiaRight':'pup_diam_r',}
    df=df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    df = df[idx]

    # drop useless gaze frames
    df = df.dropna(axis=0,subset='timestamp')

    # set timestamps t0 to start of video, convert from s to ms and set as index
    df.loc[:,'timestamp'] -= t0
    df.loc[:,'timestamp'] *= 1000.0
    df = df.set_index('timestamp')

    # return the dataframe
    return df