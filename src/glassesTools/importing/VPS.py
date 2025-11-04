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
import datetime

from ..recording import Recording
from ..eyetracker import EyeTracker
from .. import naming, timestamps, video_utils


def preprocessData(output_dir: str|pathlib.Path=None, device: str|EyeTracker=None, source_dir: str|pathlib.Path=None, rec_info: Recording=None, copy_scene_video = True, source_dir_as_relative_path = False, cam_cal_file: str|pathlib.Path=None) -> Recording:
    from . import check_folders, check_device, _store_data
    device, rec_info, _ = check_device(device, rec_info)
    if not device in [EyeTracker.VPS_19, EyeTracker.VPS_Lite]:
        raise ValueError(f'Provided device ({rec_info.eye_tracker.value}) is not a {EyeTracker.VPS_19.value} or a {EyeTracker.VPS_Lite.value}.')
    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, device)
    print(f'processing: {source_dir.name} -> {output_dir}')


    ### check and copy needed files to the output directory
    print('  Check and copy raw data...')
    ### check tobii recording and get export directory
    if rec_info is not None:
        checkRecording(device, source_dir, rec_info)
    else:
        rec_info = getRecordingInfo(source_dir)
        if rec_info is None:
            raise RuntimeError(f"The folder {source_dir} is not recognized as a {device.value} recording.")

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()


    ### copy the raw data to the output directory
    copyVPSRecording(device, source_dir, output_dir, rec_info, copy_scene_video)

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


def getRecordingInfo(inputDir: str|pathlib.Path, device: EyeTracker) -> Recording:
    # get recordings. A folder can contain multiple recordings.
    inputDir = pathlib.Path(inputDir)

    if device==EyeTracker.VPS_19:
        vid_exts = ['.mkv','.mp4']
    elif device==EyeTracker.VPS_Lite:
        vid_exts = ['.mp4']
    # recordings are identified as a tsv and an mkv file with the
    # same name
    recInfos: list[Recording] = []
    for r in inputDir.glob('*.tsv'):
        if not any(r.with_suffix(ext).is_file() for ext in vid_exts):
            continue
        # get more info
        with open(r,'rt') as f:
            lines = []
            for _ in range(5):
                lines.append(f.readline())
        uInfo = json.loads(lines[4].removeprefix('# system').removeprefix(':'))  # info about the system, remove : separately as it seems only VPS 19 and not VPS Lite has it
        # a VPS 19 recording will have a 'Smart Unit' entry
        if device==EyeTracker.VPS_19 and 'Smart Unit' not in uInfo:
            continue
        # a VPS Lite recording will have a 'Lite Unit' entry
        if device==EyeTracker.VPS_Lite and 'Lite Unit' not in uInfo:
            continue
        recInfos.append(Recording(source_directory=inputDir, eye_tracker=device))
        recInfos[-1].name = r.stem
        time_string = lines[2][len('# Recording start: '):].strip()
        recInfos[-1].start_time = timestamps.Timestamp(int(datetime.datetime.fromisoformat(time_string).timestamp()))
        recInfos[-1].glasses_serial = uInfo['glasses']
        recInfos[-1].recording_unit_serial = uInfo['Smart Unit'] if device==EyeTracker.VPS_19 else uInfo['Lite Unit']
        recInfos[-1].firmware_version = uInfo['operating system']

    # should return None if no valid recordings found
    return recInfos if recInfos else None


def checkRecording(device: EyeTracker, inputDir: str|pathlib.Path, recInfo: Recording):
    # check we have an exported gaze data file
    inputDir = pathlib.Path(inputDir)

    if device==EyeTracker.VPS_19:
        vid_exts = ['.mkv','.mp4']
    elif device==EyeTracker.VPS_Lite:
        vid_exts = ['.mp4']

    def _raise_if_file_doesnt_exist(file: str|pathlib.Path):
        if not (inputDir / file).is_file():
            raise RuntimeError(f'Recording {recInfo.name} not found: {file} file not found in {inputDir}.')
    _raise_if_file_doesnt_exist(f'{recInfo.name}.tsv')
    if not any((inputDir / f'{recInfo.name}{ext}').is_file() for ext in vid_exts):
        raise RuntimeError(f'Recording {recInfo.name} not found: no {recInfo.name}.ext video file found in {inputDir} where ext is one of the expected extensions: {vid_exts}.')

    return True

def copyVPSRecording(device: EyeTracker, inputDir: pathlib.Path, outputDir: pathlib.Path, recInfo: Recording, copy_scene_video: bool):
    """
    Copy the relevant files from the specified input dir to the specified output dir
    """
    # Copy relevant files to new directory
    if device==EyeTracker.VPS_19:
        srcFile  = inputDir / f'{recInfo.name}.mkv'
        if not srcFile.is_file():
            srcFile  = inputDir / f'{recInfo.name}.mp4'
    elif device==EyeTracker.VPS_Lite:
        srcFile  = inputDir / f'{recInfo.name}.mp4'

    if copy_scene_video:
        destFile = outputDir / f'{naming.scene_camera_video_fname_stem}{srcFile.suffix}'
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