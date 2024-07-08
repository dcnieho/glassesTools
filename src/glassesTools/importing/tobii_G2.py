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
import struct
import math
import datetime

from ..recording import Recording
from ..eyetracker import EyeTracker
from .. import timestamps, video_utils


def preprocessData(output_dir: str|pathlib.Path=None, source_dir: str|pathlib.Path=None, rec_info: Recording=None, copy_scene_video = True) -> Recording:
    from . import check_folders, _store_data
    """
    Run all preprocessing steps on tobii data and store in output_dir
    """
    output_dir, source_dir, rec_info = check_folders(output_dir, source_dir, rec_info, EyeTracker.Tobii_Glasses_2)
    print(f'processing: {source_dir.name} -> {output_dir}')


    ### check and copy needed files to the output directory
    print('  Check and copy raw data...')
    ### check tobii recording and get export directory
    if rec_info is not None:
        checkRecording(source_dir, rec_info)
    else:
        rec_info = getRecordingInfo(source_dir)
        if rec_info is None:
            raise RuntimeError(f"The folder {source_dir} is not recognized as a {EyeTracker.Tobii_Glasses_2.value} recording.")

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()


    ### copy the raw data to the output directory
    srcVid, destVid = copyTobiiRecording(source_dir, output_dir, copy_scene_video)
    if destVid:
        rec_info.scene_video_file = destVid.name
    else:
        rec_info.scene_video_file =  srcVid.parent.parent.name + '/' + srcVid.parent.name + '/' + srcVid.name

    #### prep the copied data...
    print('  Getting camera calibration...')
    sceneVideoDimensions = getCameraFromTSLV(output_dir)
    print('  Prepping gaze data...')
    gazeDf, frameTimestamps = formatGazeData(output_dir, sceneVideoDimensions, rec_info)

    _store_data(output_dir, gazeDf, frameTimestamps, rec_info)

    return rec_info


def getRecordingInfo(inputDir: str|pathlib.Path) -> Recording:
    # returns None if not a recording directory
    recInfo = Recording(source_directory=inputDir, eye_tracker=EyeTracker.Tobii_Glasses_2)

    # get participant info
    file = inputDir / 'participant.json'
    if not file.is_file():
        return None
    with open(file, 'r') as j:
        iInfo = json.load(j)
    recInfo.participant = iInfo['pa_info']['Name']

    # get recording info
    file = inputDir / 'recording.json'
    if not file.is_file():
        return None
    with open(file, 'r') as j:
        iInfo = json.load(j)
    recInfo.name = iInfo['rec_info']['Name']
    recInfo.duration   = int(iInfo['rec_length']*1000)          # in seconds, convert to ms
    time_string = iInfo['rec_created']
    if time_string[-4:].isdigit() and time_string[-5:-4]=='+':
        # add hour:minute separator for ISO 8601 format that datetime understands
        time_string = time_string[:-2]+':'+time_string[-2:]
    recInfo.start_time = timestamps.Timestamp(int(datetime.datetime.fromisoformat(time_string).timestamp()))

    # get system info
    file = inputDir / 'sysinfo.json'
    if not file.is_file():
        return None
    with open(file, 'r') as j:
        iInfo = json.load(j)

    recInfo.firmware_version = iInfo['servicemanager_version']
    recInfo.glasses_serial = iInfo['hu_serial']
    recInfo.recording_unit_serial = iInfo['ru_serial']

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
    inputDir = inputDir / 'segments' / '1'
    srcFile  = inputDir / 'fullstream.mp4'
    if copy_scene_video:
        destFile = outputDir / 'worldCamera.mp4'
        shutil.copy2(str(srcFile), str(destFile))
    else:
        destFile = None

    # Unzip the gaze data and tslv files
    for f in ['livedata.json.gz', 'et.tslv.gz']:
        with gzip.open(str(inputDir / f)) as zipFile:
            with open(outputDir / pathlib.Path(f).stem, 'wb') as unzippedFile:
                shutil.copyfileobj(zipFile, unzippedFile)

    return srcFile, destFile

def getCameraFromTSLV(inputDir: str|pathlib.Path):
    """
    Read binary TSLV file until camera calibration information is retrieved
    """
    with open(str(inputDir / 'et.tslv'), "rb") as f:
        # first look for camera item (TSLV type==300)
        while True:
            tslvType= struct.unpack('h',f.read(2))[0]
            status  = struct.unpack('h',f.read(2))[0]
            payloadLength = struct.unpack('i',f.read(4))[0]
            payloadLengthPadded = math.ceil(payloadLength/4)*4
            if tslvType != 300:
                # skip payload
                f.read(payloadLengthPadded)
            else:
                break

        # read info about camera
        camera = {}
        camera['id']       = struct.unpack('b',f.read(1))[0]
        camera['location'] = struct.unpack('b',f.read(1))[0]
        f.read(2) # skip padding
        camera['position'] = np.array(struct.unpack('3f',f.read(4*3)))
        camera['rotation'] = np.reshape(struct.unpack('9f',f.read(4*9)),(3,3))
        camera['focalLength'] = np.array(struct.unpack('2f',f.read(4*2)))
        camera['skew'] = struct.unpack('f',f.read(4))[0]
        camera['principalPoint'] = np.array(struct.unpack('2f',f.read(4*2)))
        camera['radialDistortion'] = np.array(struct.unpack('3f',f.read(4*3)))
        camera['tangentialDistortion'] = np.array(struct.unpack('3f',f.read(4*3))[:-1]) # drop last element (always zero), since there are only two tangential distortion parameters
        camera['resolution'] = np.array(struct.unpack('2h',f.read(2*2)))

    # turn into camera matrix and distortion coefficients as used by OpenCV
    camera['cameraMatrix'] = np.identity(3)
    camera['cameraMatrix'][0,0] = camera['focalLength'][0]
    camera['cameraMatrix'][0,1] = camera['skew']
    camera['cameraMatrix'][1,1] = camera['focalLength'][1]
    camera['cameraMatrix'][0,2] = camera['principalPoint'][0]
    camera['cameraMatrix'][1,2] = camera['principalPoint'][1]

    camera['distCoeff'] = np.zeros(5)
    camera['distCoeff'][:2]  = camera['radialDistortion'][:2]
    camera['distCoeff'][2:4] = camera['tangentialDistortion'][:2]
    camera['distCoeff'][4]   = camera['radialDistortion'][2]


    # store to file
    fs = cv2.FileStorage(inputDir / 'calibration.xml', cv2.FILE_STORAGE_WRITE)
    for key,value in camera.items():
        fs.write(name=key,val=value)
    fs.release()

    # tslv no longer needed, remove
    (inputDir / 'et.tslv').unlink(missing_ok=True)

    return camera['resolution']


def formatGazeData(inputDir: str|pathlib.Path, sceneVideoDimensions: list[int], recInfo: Recording):
    """
    load livedata.json
    format to get the gaze coordinates w/r/t world camera, and timestamps for every frame of video

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps
    """

    # convert the json file to pandas dataframe
    df,scene_video_ts_offset = json2df(inputDir / 'livedata.json', sceneVideoDimensions)

    # read video file, create array of frame timestamps
    frameTimestamps = video_utils.getFrameTimestampsFromVideo(recInfo.get_scene_video_path())
    frameTimestamps['timestamp'] += scene_video_ts_offset

    # use the frame timestamps to assign a frame number to each data point
    frameIdx = video_utils.tssToFrameNumber(df.index,frameTimestamps['timestamp'].to_numpy())
    df.insert(0,'frame_idx',frameIdx['frame_idx'])

    # build the formatted dataframe
    df.index.name = 'timestamp'

    # return the gaze data df and frame time stamps array
    return df, frameTimestamps


def json2df(jsonFile: str|pathlib.Path, sceneVideoDimensions: list[int]) -> tuple[pd.DataFrame, float]:
    """
    convert the livedata.json file to a pandas dataframe
    """
    # dicts to store sync points
    vtsSync  = list()       # scene video timestamp sync
    evtsSync = list()       # eye video timestamp sync (only if eye video was recorded)
    df = pd.DataFrame()     # empty dataframe to write data to

    with open(jsonFile, 'r') as file:
        entries = json.loads('[' + file.read().replace('\n', ',')[:-1] + ']')

    # loop over all lines in json file, each line represents unique json object
    for entry in entries:
        # if non-zero status (error), ensure data found in packet is marked as missing
        isError = False
        if entry['s'] != 0:
            isError = True

        ### a number of different dictKeys are possible, respond accordingly
        if 'vts' in entry.keys(): # "vts" key signfies a scene video timestamp (first frame, first keyframe, and ~1/min afterwards)
            vtsSync.append((entry['ts'], entry['vts'] if not isError else math.nan))
            continue

        ### a number of different dictKeys are possible, respond accordingly
        if 'evts' in entry.keys(): # "evts" key signfies an eye video timestamp (first frame, first keyframe, and ~1/min afterwards)
            evtsSync.append((entry['ts'], entry['evts'] if not isError else math.nan))
            continue

        # if this json object contains "eye" data (e.g. pupil info)
        if 'eye' in entry.keys():
            which_eye = entry['eye'][:1]
            if 'pc' in entry.keys():
                # origin of gaze vector is the pupil center
                df.loc[entry['ts'], 'gaze_ori_'+which_eye+'_x'] = entry['pc'][0] if not isError else math.nan
                df.loc[entry['ts'], 'gaze_ori_'+which_eye+'_y'] = entry['pc'][1] if not isError else math.nan
                df.loc[entry['ts'], 'gaze_ori_'+which_eye+'_z'] = entry['pc'][2] if not isError else math.nan
            elif 'pd' in entry.keys():
                df.loc[entry['ts'], 'pup_diam_'+which_eye] = entry['pd'] if not isError else math.nan
            elif 'gd' in entry.keys():
                df.loc[entry['ts'], 'gaze_dir_'+which_eye+'_x'] = entry['gd'][0] if not isError else math.nan
                df.loc[entry['ts'], 'gaze_dir_'+which_eye+'_y'] = entry['gd'][1] if not isError else math.nan
                df.loc[entry['ts'], 'gaze_dir_'+which_eye+'_z'] = entry['gd'][2] if not isError else math.nan

        # otherwise it contains gaze position data
        else:
            if 'gp' in entry.keys():
                df.loc[entry['ts'], 'gaze_pos_vid_x'] = entry['gp'][0]*sceneVideoDimensions[0] if not isError else math.nan
                df.loc[entry['ts'], 'gaze_pos_vid_y'] = entry['gp'][1]*sceneVideoDimensions[1] if not isError else math.nan
            elif 'gp3' in entry.keys():
                df.loc[entry['ts'], 'gaze_pos_3d_x'] = entry['gp3'][0] if not isError else math.nan
                df.loc[entry['ts'], 'gaze_pos_3d_y'] = entry['gp3'][1] if not isError else math.nan
                df.loc[entry['ts'], 'gaze_pos_3d_z'] = entry['gp3'][2] if not isError else math.nan

        # ignore anything else

    # find out t0. Do the same as GlassesViewer so timestamps are compatible
    # that is t0 is at timestamp of last video start (scene or eye)
    vtsSync  = np.array( vtsSync)
    evtsSync = np.array(evtsSync)
    t0s = [vtsSync[vtsSync[:,1]==0,0]]
    if len(evtsSync)>0:
        t0s.append(evtsSync[evtsSync[:,1]==0,0])
    t0 = max(t0s)

    # get timestamp offset for scene video
    scene_video_ts_offset = (t0s[0]-t0) / 1000.0

    # convert timestamps from us to ms
    df.index = (df.index - t0) / 1000.0

    # json no longer needed, remove
    jsonFile.unlink(missing_ok=True)

    # return the dataframe
    return df, scene_video_ts_offset