"""
Cast raw Tobii data into common format.

The output directory will contain:
    - frameTimestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gazeData.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera,
                    and 3D coordinates in the coordinate frame of the glasses (may need rotation/translation
                    to represent in world camera's coordinate frame)
    - calibration.xml: info about the camera intrinsics and transform glasses coordinate system to
                       camera coordinate system
"""

import shutil
import pathlib
import json
import gzip
import cv2
import pandas as pd
import numpy as np
import math
import datetime

from ..recording import Recording
from ..eyetracker import EyeTracker
from .. import data_files, timestamps, video_utils


def preprocessData(output_dir: str|pathlib.Path=None, source_dir: str|pathlib.Path=None, rec_info: Recording=None, copy_scene_video = True) -> Recording:
    from . import check_folders, _store_data
    """
    Run all preprocessing steps on tobii data and store in output_dir
    """
    output_dir, source_dir, rec_info = check_folders(output_dir, source_dir, rec_info, EyeTracker.Tobii_Glasses_3)
    print(f'processing: {source_dir.name} -> {output_dir}')


    ### check and copy needed files to the output directory
    print('  Check and copy raw data...')
    ### check tobii recording and get export directory
    if rec_info is not None:
        checkRecording(source_dir, rec_info)
    else:
        rec_info = getRecordingInfo(source_dir)
        if rec_info is None:
            raise RuntimeError(f"The folder {source_dir} is not recognized as a {EyeTracker.Tobii_Glasses_3.value} recording.")

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()


    ### copy the raw data to the output directory
    srcVid, destVid = copyTobiiRecording(source_dir, output_dir, copy_scene_video)
    if destVid:
        rec_info.scene_video_file = destVid.name
    else:
        rec_info.scene_video_file =  srcVid.name

    #### prep the copied data...
    print('  Getting camera calibration...')
    sceneVideoDimensions = getCameraFromJson(source_dir, output_dir)
    print('  Prepping gaze data...')
    gazeDf, frameTimestamps = formatGazeData(output_dir, sceneVideoDimensions, rec_info)

    _store_data(output_dir, gazeDf, frameTimestamps, rec_info)

    return rec_info


def getRecordingInfo(inputDir: str|pathlib.Path) -> Recording:
    # returns None if not a recording directory
    recInfo = Recording(source_directory=inputDir, eye_tracker=EyeTracker.Tobii_Glasses_3)

    # get recording info
    file = inputDir / 'recording.g3'
    if not file.is_file():
        return None
    with open(file, 'rb') as j:
        rInfo = json.load(j)
    recInfo.name = rInfo['name']
    recInfo.duration = int(rInfo['duration']*1000)          # in seconds, convert to ms
    time_string = rInfo['created']
    if time_string[-1:]=='Z':
        # change Z suffix to +00:00 for ISO 8601 format that datetime understands
        time_string = time_string[:-1]+'+00:00'
    recInfo.start_time = timestamps.Timestamp(int(datetime.datetime.fromisoformat(time_string).timestamp()))


    # get participant info (if available)
    file = inputDir / rInfo['meta-folder'] / 'participant'
    if file.is_file():
        with open(file, 'rb') as j:
            pInfo = json.load(j)
        recInfo.participant = pInfo['name']

    # get system info
    recInfo.firmware_version = (inputDir / rInfo['meta-folder'] / 'RuVersion').read_text()
    recInfo.glasses_serial = (inputDir / rInfo['meta-folder'] / 'HuSerial').read_text()
    recInfo.recording_unit_serial = (inputDir / rInfo['meta-folder'] / 'RuSerial').read_text()

    # we got a valid recording and at least some info if we got here
    # return what we've got
    return recInfo


def checkRecording(inputDir: str|pathlib.Path, recInfo: Recording):
    actualRecInfo = getRecordingInfo(inputDir)
    if actualRecInfo is None or recInfo.name!=actualRecInfo.name:
        raise ValueError(f"A recording with the name \"{recInfo.name}\" was not found in the folder {inputDir}.")

    # make sure caller did not mess with recInfo
    if recInfo.participant!=actualRecInfo.participant:
        raise ValueError(f"A recording with the participant \"{recInfo.participant}\" was not found in the folder {inputDir}.")
    if recInfo.duration!=actualRecInfo.duration:
        raise ValueError(f"A recording with the duration \"{recInfo.duration}\" was not found in the folder {inputDir}.")
    if recInfo.start_time.value!=actualRecInfo.start_time.value:
        raise ValueError(f"A recording with the start_time \"{recInfo.start_time.display}\" was not found in the folder {inputDir}.")
    if recInfo.firmware_version!=actualRecInfo.firmware_version:
        raise ValueError(f"A recording with the firmware_version \"{recInfo.firmware_version}\" was not found in the folder {inputDir}.")
    if recInfo.glasses_serial!=actualRecInfo.glasses_serial:
        raise ValueError(f"A recording with the glasses_serial \"{recInfo.glasses_serial}\" was not found in the folder {inputDir}.")
    if recInfo.recording_unit_serial!=actualRecInfo.recording_unit_serial:
        raise ValueError(f"A recording with the recording_unit_serial \"{recInfo.recording_unit_serial}\" was not found in the folder {inputDir}.")


def copyTobiiRecording(inputDir: pathlib.Path, outputDir: pathlib.Path, copy_scene_video: bool):
    """
    Copy the relevant files from the specified input dir to the specified output dir
    """
    # Copy relevent files to new directory
    srcFile  = inputDir / 'scenevideo.mp4'
    if copy_scene_video:
        destFile = outputDir / 'worldCamera.mp4'
        shutil.copy2(str(srcFile), str(destFile))
    else:
        destFile = None

    # Unzip the gaze data file
    for f in ['gazedata.gz']:
        with gzip.open(str(inputDir / f)) as zipFile:
            with open(outputDir / pathlib.Path(f).stem, 'wb') as unzippedFile:
                shutil.copyfileobj(zipFile, unzippedFile)

    return srcFile, destFile

def getCameraFromJson(inputDir: str|pathlib.Path, outputDir: str|pathlib.Path):
    """
    Read camera calibration from recording information file
    """
    with open(inputDir / 'recording.g3', 'rb') as f:
        rInfo = json.load(f)

    camera = rInfo['scenecamera']['camera-calibration']

    # rename some fields, ensure they are numpy arrays
    camera['focalLength'] = np.array(camera.pop('focal-length'))
    camera['principalPoint'] = np.array(camera.pop('principal-point'))
    camera['radialDistortion'] = np.array(camera.pop('radial-distortion'))
    camera['tangentialDistortion'] = np.array(camera.pop('tangential-distortion'))

    camera['position'] = np.array(camera['position'])
    camera['resolution'] = np.array(camera['resolution'])
    camera['rotation'] = np.array(camera['rotation'])

    # turn into camera matrix and distortion coefficients as used by OpenCV
    camera['cameraMatrix'] = np.identity(3)
    camera['cameraMatrix'][0,0] = camera['focalLength'][0]
    camera['cameraMatrix'][0,1] = camera['skew']
    camera['cameraMatrix'][1,1] = camera['focalLength'][1]
    camera['cameraMatrix'][0,2] = camera['principalPoint'][0]
    camera['cameraMatrix'][1,2] = camera['principalPoint'][1]

    camera['distCoeff'] = np.zeros(5)
    camera['distCoeff'][:2]  = camera['radialDistortion'][:2]
    camera['distCoeff'][2:4] = camera['tangentialDistortion']
    camera['distCoeff'][4]   = camera['radialDistortion'][2]


    # store to file
    fs = cv2.FileStorage(outputDir / 'calibration.xml', cv2.FILE_STORAGE_WRITE)
    for key,value in camera.items():
        fs.write(name=key,val=value)
    fs.release()

    return camera['resolution']


def formatGazeData(inputDir: str|pathlib.Path, sceneVideoDimensions: list[int], recInfo: Recording):
    """
    load gazedata json file
    format to get the gaze coordinates w/r/t world camera, and timestamps for every frame of video

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps
    """

    # convert the json file to pandas dataframe
    df = json2df(inputDir / 'gazedata', sceneVideoDimensions)

    # read video file, create array of frame timestamps
    frameTimestamps = video_utils.getFrameTimestampsFromVideo(recInfo.get_scene_video_path())

    # use the frame timestamps to assign a frame number to each data point
    frameIdx = video_utils.tssToFrameNumber(df.index,frameTimestamps['timestamp'].to_numpy())
    df.insert(0,'frame_idx',frameIdx['frame_idx'])

    # return the gaze data df and frame time stamps array
    return df, frameTimestamps


def json2df(jsonFile: str|pathlib.Path, sceneVideoDimensions: list[int]) -> pd.DataFrame:
    """
    convert the livedata.json file to a pandas dataframe
    """

    with open(jsonFile, 'r') as file:
        entries = json.loads('[' + file.read().replace('\n', ',')[:-1] + ']')

    # json no longer needed, remove
    jsonFile.unlink(missing_ok=True)

    # turn gaze data into data frame
    dfR = pd.json_normalize(entries)
    # convert timestamps from s to ms and set as index
    dfR.loc[:,'timestamp'] *= 1000.0
    dfR = dfR.set_index('timestamp')
    # drop anything thats not gaze
    dfR = dfR.drop(dfR[dfR.type != 'gaze'].index)
    # manipulate data frame to expand columns as needed
    df = pd.DataFrame([],index=dfR.index)
    expander = lambda a,n: [[math.nan]*n if not isinstance(x,list) else x for x in a]
    # monocular gaze data
    for eye in ('left','right'):
        if 'data.eye'+eye+'.gazeorigin' not in dfR.columns:
            continue    # no data at all for this eye
        which_eye = eye[:1]
        df[data_files.getColumnLabels('gaze_ori_'+which_eye,3)] = pd.DataFrame(expander(dfR['data.eye'+eye+'.gazeorigin'   ].tolist(),3), index=dfR.index)
        df[data_files.getColumnLabels('gaze_dir_'+which_eye,3)] = pd.DataFrame(expander(dfR['data.eye'+eye+'.gazedirection'].tolist(),3), index=dfR.index)
        df['pup_diam_'+which_eye] = dfR['data.eye'+eye+'.pupildiameter']

    # binocular gaze data
    df[data_files.getColumnLabels('gaze_pos_3d',3) ] = pd.DataFrame(expander(dfR['data.gaze3d'].tolist(),3), index=dfR.index)
    df[data_files.getColumnLabels('gaze_pos_vid',2)] = pd.DataFrame(expander(dfR['data.gaze2d'].tolist(),2), index=dfR.index)
    df.loc[:,'gaze_pos_vid_x'] *= sceneVideoDimensions[0]
    df.loc[:,'gaze_pos_vid_y'] *= sceneVideoDimensions[1]

    # return the dataframe
    return df