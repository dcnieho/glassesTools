"""
Cast raw SMI ETG data into common format.

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
import os
import pathlib
import cv2
import pandas as pd
import numpy as np
import math
import configparser
from io import StringIO
from scipy.spatial.transform import Rotation

from ..recording import Recording
from ..eyetracker import EyeTracker
from .. import video_utils


def preprocessData(output_dir: str|pathlib.Path, source_dir: str|pathlib.Path=None, rec_info: Recording=None, copy_scene_video = True) -> Recording:
    from . import check_folders, _store_data
    """
    Run all preprocessing steps on SMI data and store in output_dir
    """
    # NB: copy_scene_video input argument might be ignored. If ffmpeg is present, it will be used to transcode the scene camera video
    output_dir, source_dir, rec_info = check_folders(output_dir, source_dir, rec_info, EyeTracker.SMI_ETG)
    print(f'processing: {source_dir.name} -> {output_dir}')


    ### check and copy needed files to the output directory
    print('  Check and copy raw data...')
    if rec_info is not None:
        if not checkRecording(source_dir, rec_info, use_return=True):
            raise ValueError(f"A recording with the name \"{rec_info.name}\" was not found in the folder {source_dir}. Check that the name is correct and make sure that you export the scene video and gaze data using BeGaze as described in the glassesValidator manual.")
    else:
        recInfos = getRecordingInfo(source_dir)
        if recInfos is None:
            raise RuntimeError(f"The folder {source_dir} does not contain SMI ETG recordings prepared for glassesValidator. If this is an SMI recording folder, you may not have run the required gaze data and scene video exports yet. See the glassesValidator manual for which exports you should perform with BeGaze first as well as the file naming scheme.")
        rec_info = recInfos[0]  # take first, arbitrarily. If anything else wanted, user should call this function with a correct rec_info themselves

    # make output dirs
    if not output_dir.is_dir():
        output_dir.mkdir()


    ### copy the raw data to the output directory
    checkRecording(source_dir, rec_info)
    copySMIRecordings(source_dir, output_dir, rec_info, copy_scene_video)

    #### prep the data
    print('  Getting camera calibration...')
    sceneVideoDimensions = getCameraFromFile(source_dir, output_dir)
    print('  Prepping gaze data...')
    gazeDf, frameTimestamps = formatGazeData(source_dir, rec_info, sceneVideoDimensions)

    _store_data(output_dir, gazeDf, frameTimestamps, rec_info)

    return rec_info

def getRecordingInfo(inputDir: str|pathlib.Path) -> list[Recording]:
    # returns None if directory does not any recordings

    # NB: can be multiple recordings in an SMI folder

    # but first see this is a recording folder at all
    if not (inputDir / 'codec1.bin').is_file():
        return None
    camInfo = readSMICamInfoFile(inputDir)
    serial = camInfo.get("MiiSimulation",'DeviceSerialNumber')

    # get recordings. We expect the user to rename their exports to have the same format
    # as the other files in a project directory. So e.g., data exported from 001-2-recording.idf
    # for the corresponding 001-2-recording.avi, should be named 001-2-recording.txt. The
    # exported video should be called 001-2-export.avi
    recInfos = []
    for r in inputDir.glob('*-export.avi'):
        recInfos.append(Recording(source_directory=inputDir, eye_tracker=EyeTracker.SMI_ETG))
        recInfos[-1].participant = inputDir.name
        recInfos[-1].name = str(r.name)[:-len('-export.avi')]
        recInfos[-1].glasses_serial = serial

    # should return None if no valid recordings found
    return recInfos if recInfos else None


def checkRecording(inputDir: str|pathlib.Path, recInfo: str|pathlib.Path, use_return = False):
    """
    This checks that the folder is properly prepared
    (i.e. the required BeGaze exports were run)
    """
    # check we have an exported gaze data file
    file = recInfo.name + '-export.avi'
    if not (inputDir / file).is_file():
        if use_return:
            return False
        else:
            raise RuntimeError(f'{file} file not found in {inputDir}. Make sure you export the scene video using BeGaze as described in the glassesValidator manual.')

    # check we have an exported scene video
    file = recInfo.name + '-recording.txt'
    if not (inputDir / file).is_file():
        if use_return:
            return False
        else:
            raise RuntimeError(f'{file} file not found in {inputDir}. Make sure you export the gaze data using BeGaze as described in the glassesValidator manual.')

    return True


def copySMIRecordings(inputDir: pathlib.Path, outputDir: pathlib.Path, recInfo: Recording, copy_scene_video: bool):
    """
    Copy the relevant files from the specified input dir to the specified output dirs
    """

    # Copy relevent files to new directory
    file    = recInfo.name + '-export.avi'
    srcFile = inputDir / file

    # if ffmpeg is on path, remux avi to mp4 (reencode audio from flac to aac as flac is not supported in mp4)
    # else just copy. Ignore copy_scene_video in this case
    if shutil.which('ffmpeg') is not None:
        # make mp4
        destFile = outputDir / 'worldCamera.mp4'
        cmd_str = ' '.join(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', '"'+str(srcFile)+'"', '-vcodec', 'copy', '-acodec', 'aac', '"'+str(destFile)+'"'])
        os.system(cmd_str)
    else:
        if copy_scene_video:
            destFile = outputDir / 'worldCamera.avi'
            shutil.copy2(str(srcFile), str(destFile))
        else:
            destFile = None

    if destFile:
        recInfo.scene_video_file = destFile.name
    else:
        recInfo.scene_video_file =  srcFile.name


def readSMICamInfoFile(inputDir: str|pathlib.Path):
    with open(inputDir / 'codec1.bin', 'r') as file:
        camInfoStr = file.read().replace('## ', '')

    camInfo = configparser.ConfigParser(converters={'nparray': lambda x: np.fromstring(x,sep='\t')})
    camInfo.read_string(camInfoStr)
    return camInfo


def getCameraFromFile(inputDir: str|pathlib.Path, outputDir: str|pathlib.Path):
    """
    Read camera calibration from information file
    """
    camInfo = readSMICamInfoFile(inputDir)

    camera = {}
    camera['FOV'] = camInfo.getfloat("MiiSimulation",'SceneCamFOV')
    camera['resolution'] = np.array([camInfo.getint("MiiSimulation",'SceneCamWidth'), camInfo.getint("MiiSimulation",'SceneCamHeight')])
    camera['sensorOffsets'] = np.array([camInfo.getfloat("MiiSimulation",'SceneCamSensorOffsetX'), camInfo.getfloat("MiiSimulation",'SceneCamSensorOffsetY')])

    camera['radialDistortion'] = camInfo.getnparray("MiiSimulation",'SceneCamRadialDistortion')
    camera['tangentialDistortion'] = camInfo.getnparray("MiiSimulation",'SceneCamTangentialDistortion')

    camera['position'] = camInfo.getnparray("MiiSimulation",'SceneCamPos')
    camera['eulerAngles'] = np.array([camInfo.getfloat("MiiSimulation",'SceneCamOrX'), camInfo.getfloat("MiiSimulation",'SceneCamOrY'), camInfo.getfloat("MiiSimulation",'SceneCamOrZ')])

    # now turn these fields into focal length and principal point
    # 1. FOV is horizontal FOV of camera, given resolution we can compute
    # focal length. We assume vertical focal length is the same, best we can do
    fl = camera['resolution'][0]/(2*math.tan(camera['FOV']/2/180*math.pi))
    camera['focalLength'] = np.array([fl, fl])
    # 2. sensor offsets seem to be relative to center of sensor
    camera['principalPoint'] = camera['resolution']/2.+camera['sensorOffsets']
    # 3. turn euler angles into rotation matrix (180-Rz because poster space has positive X rightward and positive Y downward)
    camera['rotation'] = Rotation.from_euler('XYZ', [camera['eulerAngles'][0], camera['eulerAngles'][1], 180-camera['eulerAngles'][2]], degrees=True).as_matrix()

    # turn into camera matrix and distortion coefficients as used by OpenCV
    camera['cameraMatrix'] = np.identity(3)
    camera['cameraMatrix'][0,0] = camera['focalLength'][0]
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


def formatGazeData(inputDir: str|pathlib.Path, recInfo: Recording, sceneVideoDimensions: list[int]):
    """
    load gazedata file
    format to get the gaze coordinates w.r.t. world camera, and timestamps for
    every frame of video

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps
    """

    # convert the text file to pandas dataframe
    file = recInfo.name + '-recording.txt'
    df = gazedata2df(inputDir / file, sceneVideoDimensions)

    # read video file, create array of frame timestamps
    frameTimestamps = video_utils.getFrameTimestampsFromVideo(recInfo.get_scene_video_path())

    # SMI frame counter seems to be of the format HH:MM:SS:FR, where HH:MM:SS is a normal
    # hour, minute, second timecode, and FR is a frame number for within that second. The
    # frame number is zero-based, and runs ranges from 0 to FS-1 where FS is the frame
    # rate of the camera (e.g. 24 Hz). Convert their frame counter to a normal counter
    df_fr = pd.DataFrame(df['Frame'].apply(lambda x: [int(y) for y in x.split(':')]).to_list(),columns=['hour','minute','second','frame'])
    frameRate = df_fr.frame.max()+1
    df.insert(0,'frame_idx',(((df_fr.hour*60 + df_fr.minute)*60 + df_fr.second)*frameRate + df_fr.frame).to_numpy())
    # NB: seems we can get frame numbers in the data which are beyond the length of the video. so be it
    df=df.drop(columns=['Frame'])
    # NB: it seems the SMI export doesn't strictly follow their own timecode, but uses the first
    # gaze data point for the first frame of the video, which is then also timestamped with the timecode
    # of that first frame. Subtracting min so that the frame_idx starts at 0 for the data also empirically
    # seems to line up with the SMI export (most of the time, sadly seems to vary a little between videos)
    df.frame_idx = df.frame_idx-df.frame_idx.min()

    # return the gaze data df and frame time stamps array
    return df, frameTimestamps


def gazedata2df(textFile,sceneVideoDimensions):
    """
    convert the gazedata file to a pandas dataframe
    """

    with open(textFile, 'r') as file:
        textData = file.read()

    df = pd.read_table(StringIO(textData),comment='#',index_col=False)

    # get assumed viewing distance
    matchedLines    = [line for line in textData.split('\n') if "Head Distance [mm]" in line]
    viewingDistance = float(matchedLines[0].split('\t')[1])

    # prepare data frame
    # remove unneeded columns
    df=df.drop(columns=['Type', 'Trial', 'Aux1'],errors='ignore') # drop these columns if they exist
    # if we have monocular point of regard data, check if its not identical to binocular
    if 'L POR X [px]' in df.columns:
        xEqual = np.all( np.logical_or(df['L POR X [px]'] == df['B POR X [px]'], np.isnan(df['L POR X [px]']) ))
        yEqual = np.all( np.logical_or(df['L POR Y [px]'] == df['B POR Y [px]'], np.isnan(df['L POR Y [px]']) ))
        if xEqual and yEqual:
            df=df.drop(columns=['L POR X [px]', 'L POR Y [px]'])
    if 'R POR X [px]' in df.columns:
        xEqual = np.all( np.logical_or(df['R POR X [px]'] == df['B POR X [px]'], np.isnan(df['R POR X [px]']) ))
        yEqual = np.all( np.logical_or(df['R POR Y [px]'] == df['B POR Y [px]'], np.isnan(df['R POR Y [px]']) ))
        if xEqual and yEqual:
            df=df.drop(columns=['R POR X [px]', 'R POR Y [px]'])

    # rename and reorder columns
    lookup = {'Time': 'timestamp',
               'L EPOS X':'gaze_ori_l_x',
               'L EPOS Y':'gaze_ori_l_y',
               'L EPOS Z':'gaze_ori_l_z',
               'L Pupil Diameter [mm]':'pup_diam_l',
               'L GVEC X':'gaze_dir_l_x',
               'L GVEC Y':'gaze_dir_l_y',
               'L GVEC Z':'gaze_dir_l_z',
               'R EPOS X':'gaze_ori_r_x',
               'R EPOS Y':'gaze_ori_r_y',
               'R EPOS Z':'gaze_ori_r_z',
               'R Pupil Diameter [mm]':'pup_diam_r',
               'R GVEC X':'gaze_dir_r_x',
               'R GVEC Y':'gaze_dir_r_y',
               'R GVEC Z':'gaze_dir_r_z',
               'B POR X [px]':'gaze_pos_vid_x',
               'B POR Y [px]':'gaze_pos_vid_y',
               'L Dia X [px]':'pup_diam_l_px_x',
               'L Dia Y [px]':'pup_diam_l_px_y',
               'R Dia X [px]':'pup_diam_r_px_x',
               'R Dia Y [px]':'pup_diam_r_px_y'}
    df=df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    idx.extend([x for x in df.columns if x not in idx])   # append columns not in lookup
    df = df[idx]

    # convert timestamps from us to ms and set as index
    df.loc[:,'timestamp'] /= 1000.0
    df = df.set_index('timestamp')

    # return the dataframe
    return df