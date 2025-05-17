"""
Cast Argus Science ETVision data into common format.

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


def preprocessData(output_dir: str|pathlib.Path=None, source_dir: str|pathlib.Path=None, rec_info: Recording=None, copy_scene_video = True, source_dir_as_relative_path = False, cam_cal_file: str|pathlib.Path=None) -> Recording:
    from . import check_folders, _store_data
    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, EyeTracker.Argus_ETVision)
    print(f'processing: {source_dir.name} -> {output_dir}')


    ### check and copy needed files to the output directory
    print('  Check and copy raw data...')
    ### check tobii recording and get export directory
    if rec_info is not None:
        checkRecording(source_dir, rec_info)
    else:
        rec_info = getRecordingInfo(source_dir)
        if rec_info is None:
            raise RuntimeError(f"The folder {source_dir} is not recognized as a {EyeTracker.Argus_ETVision.value} recording.")

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()


    ### copy the raw data to the output directory
    copyETVisionRecording(source_dir, output_dir, rec_info, copy_scene_video)

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
    # get recordings. If i understand correctly, a folder can contain multiple recordings.
    inputDir = pathlib.Path(inputDir)
    # recordings are identified as a tsv and an mkv file with the
    # same name
    recInfos: list[Recording] = []
    for r in inputDir.glob('*.csv'):
        if not r.with_name(f'{r.stem}_Scene.wmv').is_file():
            continue
        recInfos.append(Recording(source_directory=inputDir, eye_tracker=EyeTracker.Argus_ETVision))
        recInfos[-1].name = r.stem
        # get more info from first line
        with open(r,'rt') as f:
            line = f.readline()
        for s in line.strip().split(','):
            if 'ETVision' in s:
                recInfos[-1].firmware_version = s[len('ETVision: '):].strip()
            elif 'Start_Recording' in s:
                time_string = s[len('Start_Recording: '):].strip()
                recInfos[-1].start_time = timestamps.Timestamp(int(datetime.datetime.fromisoformat(time_string).timestamp()))

    # should return None if no valid recordings found
    return recInfos if recInfos else None


def checkRecording(inputDir: str|pathlib.Path, recInfo: Recording):
    # check we have the expected file
    for suff in ('.csv','_Scene.wmv'):
        file = f'{recInfo.name}{suff}'
        if not (inputDir / file).is_file():
            raise RuntimeError(f'Recording {recInfo.name} not found: {file} file not found in {inputDir}.')

    return True

def copyETVisionRecording(inputDir: pathlib.Path, outputDir: pathlib.Path, recInfo: Recording, copy_scene_video: bool):
    """
    Copy the relevant files from the specified input dir to the specified output dir
    """
    # Copy relevant files to new directory
    srcFile  = inputDir / f'{recInfo.name}_Scene.wmv'

    if copy_scene_video:
        destFile = outputDir / f'{naming.scene_camera_video_fname_stem}.wmv'
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
    df = gaze2df(inputDir / f'{recInfo.name}.csv')

    # read video file, create array of frame timestamps
    frameTimestamps = video_utils.get_frame_timestamps_from_video(recInfo.get_scene_video_path())

    # use the frame timestamps to assign a frame number to each data point
    frameIdx = video_utils.timestamps_to_frame_number(df.index,frameTimestamps['timestamp'].to_numpy())
    df.insert(0,'frame_idx',frameIdx['frame_idx'])

    # return the gaze data df and frame time stamps array
    return df, frameTimestamps


def gaze2df(gazeFile: str|pathlib.Path) -> pd.DataFrame:
    """
    convert the .tsv file to a pandas dataframe
    """

    df = pd.read_csv(gazeFile,index_col=False,skiprows=1)

    # rename and reorder columns
    # TODO: verg_gaze_coord_x verg_gaze_coord_y verg_gaze_coord_z left_eye_location_x right_eye_location_x left_eye_location_y right_eye_location_y left_eye_location_z right_eye_location_z left_gaze_dir_x right_gaze_dir_x left_gaze_dir_y right_gaze_dir_y left_gaze_dir_z right_gaze_dir_z
    lookup = {'start_of_record': 'timestamp',
               'horz_gaze_coord':'gaze_pos_vid_x',
               'vert_gaze_coord':'gaze_pos_vid_y',
               'left_pupil_diam':'pup_diam_l',
               'right_pupil_diam':'pup_diam_r',}
    df=df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    df = df[idx]

    # convert timestamps from s to ms and set as index
    df.loc[:,'timestamp'] *= 1000.0
    df = df.set_index('timestamp')

    # return the dataframe
    return df