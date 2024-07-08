"""
Cast raw pupil labs (Core, Invisible and Neon) data into common format.

The output directory will contain:
    - frameTimestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gazeData.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera,
                    and 3D coordinates in the coordinate frame of the glasses (may need rotation/translation
                    to represent in world camera's coordinate frame)
    - calibration.xml: info about the camera intrinsics
"""

import shutil
import pathlib
import json
import cv2
import pandas as pd
import numpy as np
import msgpack
import re
import typing
from urllib.request import urlopen

from ..recording import Recording
from ..eyetracker import EyeTracker
from .. import timestamps, video_utils


def preprocessData(output_dir: str|pathlib.Path, device: str|EyeTracker=None, source_dir: str|pathlib.Path=None, rec_info: Recording=None, copy_scene_video = True) -> Recording:
    from . import check_folders, check_device, _store_data
    """
    Run all preprocessing steps on pupil data and store in output_dir
    """
    device, rec_info = check_device(device, rec_info)
    if not device in [EyeTracker.Pupil_Core, EyeTracker.Pupil_Invisible, EyeTracker.Pupil_Neon]:
        raise ValueError(f'Provided device ({rec_info.eye_tracker.value}) is not a {EyeTracker.Pupil_Core.value}, a {EyeTracker.Pupil_Invisible.value} or a {EyeTracker.Pupil_Neon.value}.')
    output_dir, source_dir, rec_info = check_folders(output_dir, source_dir, rec_info, device)
    print(f'processing: {source_dir.name} -> {output_dir}')


    ### check and copy needed files to the output directory
    print('  Check and copy raw data...')
    ### check pupil recording and get export directory
    exportFile, is_cloud_export = checkPupilRecording(source_dir)
    if rec_info is not None:
        checkRecording(source_dir, rec_info)
    else:
        rec_info = getRecordingInfo(source_dir, device)
        if rec_info is None:
            raise RuntimeError(f"The folder {source_dir} is not recognized as a {device.value} recording.")

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()


    # find world video and copy if wanted
    if is_cloud_export:
        scene_vid = list(source_dir.glob('*.mp4'))
        if len(scene_vid)!=1:
            raise RuntimeError(f'Scene video missing or more than one found for Pupil Cloud export in folder {source_dir}')
        srcVid = scene_vid[0]
    else:
        srcVid = source_dir / 'world.mp4'
    if copy_scene_video:
        destVid = output_dir / 'worldCamera.mp4'
        shutil.copy2(str(srcVid), str(destVid))
        rec_info.scene_video_file = destVid.name
    else:
        rec_info.scene_video_file =  srcVid.name
    print(f'  Input data copied to: {output_dir}')


    ### get camera cal
    print('  Getting camera calibration...')
    if is_cloud_export:
        sceneVideoDimensions = getCameraCalFromCloudExport(source_dir, output_dir, rec_info)
    else:
        match rec_info.eye_tracker:
            case EyeTracker.Pupil_Core:
                sceneVideoDimensions = getCameraFromMsgPack(source_dir, output_dir)
            case EyeTracker.Pupil_Invisible:
                if (source_dir/'calibration.bin').is_file():
                    sceneVideoDimensions = getCameraCalFromBinFile(source_dir, output_dir, rec_info)
                else:
                    sceneVideoDimensions = getCameraCalFromOnline(source_dir, output_dir, rec_info)


    ### get gaze data and video frame timestamps
    print('  Prepping gaze data...')
    if is_cloud_export:
        gazeDf, frameTimestamps = formatGazeDataCloudExport(source_dir, exportFile)
    else:
        gazeDf, frameTimestamps = formatGazeDataPupilPlayer(source_dir, exportFile, sceneVideoDimensions, rec_info)

    _store_data(output_dir, gazeDf, frameTimestamps, rec_info)

    return rec_info


def checkPupilRecording(inputDir: str|pathlib.Path):
    """
    This checks that the folder is properly prepared,
    i.e., either:
    - opened in pupil player and an export was run (currently Pupil Core or Pupil Invisible)
    - exported from Pupil Cloud (currently Pupil Invisible or Pupil Neon)
    """
    # check we have an info.player.json file
    if not (inputDir / 'info.player.json').is_file():
        # possibly a pupil cloud export
        if not (inputDir / 'info.json').is_file() or not (inputDir / 'gaze.csv').is_file():
            raise RuntimeError(f'Neither the info.player.json file nor the info.json and gaze.csv files are found for {inputDir}. Either export from Pupil Cloud or, if the folder contains raw sensor data, open the recording in Pupil Player and run an export before importing into glassesValidator.')
        return inputDir / 'gaze.csv', True

    else:
        # check we have an export in the input dir
        inputExpDir = inputDir / 'exports'
        if not inputExpDir.is_dir():
            raise RuntimeError(f'no exports folder for {inputDir}. Perform a recording export in Pupil Player before importing into glassesValidator.')

        # get latest export in that folder that contain a gaze position file
        gpFiles = sorted(list(inputExpDir.rglob('*gaze_positions*.csv')))
        if not gpFiles:
            raise RuntimeError(f'There are no exports in the folder {inputExpDir}. Perform a recording export in Pupil Player before importing into glassesValidator.')

        return gpFiles[-1], False


def getRecordingInfo(inputDir: str|pathlib.Path, device: EyeTracker) -> Recording:
    # returns None if not a recording directory
    recInfo = Recording(source_directory=inputDir, eye_tracker=device)

    if (inputDir / 'info.player.json').is_file():
        # Pupil player export
        match device:
            case EyeTracker.Pupil_Core:
                # check this is not an invisible recording
                file = inputDir / 'info.invisible.json'
                if file.is_file():
                    return None

                file = inputDir / 'info.player.json'
                if not file.is_file():
                    return None
                with open(file, 'r') as j:
                    iInfo = json.load(j)
                recInfo.name = iInfo['recording_name']
                recInfo.start_time = timestamps.Timestamp(int(iInfo['start_time_system_s'])) # UTC in seconds, keep second part
                recInfo.duration   = int(iInfo['duration_s']*1000)                      # in seconds, convert to ms
                recInfo.recording_software_version = iInfo['recording_software_version']

                # get user name, if any
                user_info_file = inputDir / 'user_info.csv'
                if user_info_file.is_file():
                    df = pd.read_csv(user_info_file)
                    nameRow = df['key'].str.contains('name')
                    if any(nameRow):
                        if not pd.isnull(df[nameRow].value).to_numpy()[0]:
                            recInfo.participant = df.loc[nameRow,'value'].to_numpy()[0]

            case EyeTracker.Pupil_Invisible:
                file = inputDir / 'info.invisible.json'
                if not file.is_file():
                    return None
                with open(file, 'r') as j:
                    iInfo = json.load(j)
                recInfo.name = iInfo['template_data']['recording_name']
                recInfo.recording_software_version = iInfo['app_version']
                recInfo.start_time = timestamps.Timestamp(int(iInfo['start_time']//1000000000)) # UTC in nanoseconds, keep second part
                recInfo.duration   = int(iInfo['duration']//1000000)                            # in nanoseconds, convert to ms
                recInfo.glasses_serial = iInfo['glasses_serial_number']
                recInfo.recording_unit_serial = iInfo['android_device_id']
                recInfo.scene_camera_serial = iInfo['scene_camera_serial_number']
                # get participant name
                file = inputDir / 'wearer.json'
                if file.is_file():
                    wearer_id = iInfo['wearer_id']
                    with open(file, 'r') as j:
                        iInfo = json.load(j)
                    if wearer_id==iInfo['uuid']:
                        recInfo.participant = iInfo['name']

            case EyeTracker.Pupil_Neon:
                return None # there are no pupil player exports for the Neon eye tracker

            case _:
                print(f"Device {device} unknown")
                return None
    else:
        # pupil cloud export, for either Pupil Invisible or Pupil Neon
        if device==EyeTracker.Pupil_Core:
            return None

        # raw sensor data also contain an info.json (checked below), so checking
        # that file is not enough to see if this is a Cloud Export. Check gaze.csv
        # presence
        if not (inputDir / 'gaze.csv').is_file():
            return None

        file = inputDir / 'info.json'
        if not file.is_file():
            return None
        with open(file, 'r') as j:
            iInfo = json.load(j)

        # check this is for the expected device
        is_neon = 'Neon' in iInfo['android_device_name'] or 'frame_name' in iInfo
        if device==EyeTracker.Pupil_Invisible and is_neon:
            return None
        elif device==EyeTracker.Pupil_Neon and not is_neon:
            return None

        recInfo.name = iInfo['template_data']['recording_name']
        recInfo.recording_software_version = iInfo['app_version']
        recInfo.start_time = timestamps.Timestamp(int(iInfo['start_time']//1000000000))  # UTC in nanoseconds, keep second part
        recInfo.duration   = int(iInfo['duration']//1000000)                        # in nanoseconds, convert to ms
        if is_neon:
            recInfo.glasses_serial = iInfo['module_serial_number']
        else:
            recInfo.glasses_serial = iInfo['glasses_serial_number']
            recInfo.scene_camera_serial = iInfo['scene_camera_serial_number']
        recInfo.recording_unit_serial = iInfo['android_device_id']
        if is_neon:
            recInfo.firmware_version = f"{iInfo['pipeline_version']} ({iInfo['firmware_version'][0]}.{iInfo['firmware_version'][1]})"
        else:
            recInfo.firmware_version = iInfo['pipeline_version']
        recInfo.participant = iInfo['wearer_name']

    # we got a valid recording and at least some info if we got here
    # return what we've got
    return recInfo


def checkRecording(inputDir: str|pathlib.Path, recInfo: Recording):
    actualRecInfo = getRecordingInfo(inputDir, recInfo.eye_tracker)
    if actualRecInfo is None or recInfo.name!=actualRecInfo.name:
        raise ValueError(f"A recording with the name \"{recInfo.name}\" was not found in the folder {inputDir}.")

    # make sure caller did not mess with recInfo
    if recInfo.duration!=actualRecInfo.duration:
        raise ValueError(f"A recording with the duration \"{recInfo.duration}\" was not found in the folder {inputDir}.")
    if recInfo.start_time.value!=actualRecInfo.start_time.value:
        raise ValueError(f"A recording with the start_time \"{recInfo.start_time.display}\" was not found in the folder {inputDir}.")
    if recInfo.recording_software_version!=actualRecInfo.recording_software_version:
        raise ValueError(f"A recording with the duration \"{recInfo.duration}\" was not found in the folder {inputDir}.")

    # for invisible and neon recordings we have a bit more info
    if recInfo.eye_tracker in [EyeTracker.Pupil_Invisible, EyeTracker.Pupil_Neon]:
        if recInfo.glasses_serial!=actualRecInfo.glasses_serial:
            raise ValueError(f"A recording with the glasses_serial \"{recInfo.glasses_serial}\" was not found in the folder {inputDir}.")
        if recInfo.recording_unit_serial!=actualRecInfo.recording_unit_serial:
            raise ValueError(f"A recording with the recording_unit_serial \"{recInfo.recording_unit_serial}\" was not found in the folder {inputDir}.")
        if recInfo.eye_tracker==EyeTracker.Pupil_Invisible and recInfo.scene_camera_serial!=actualRecInfo.scene_camera_serial:
            raise ValueError(f"A recording with the scene_camera_serial \"{recInfo.scene_camera_serial}\" was not found in the folder {inputDir}.")
        if (recInfo.participant is not None or actualRecInfo.participant is not None) and recInfo.participant!=actualRecInfo.participant:
            raise ValueError(f"A recording with the participant \"{recInfo.participant}\" was not found in the folder {inputDir}.")


def getCameraFromMsgPack(inputDir: str|pathlib.Path, outputDir: str|pathlib.Path) -> list[int]:
    """
    Read camera calibration from recording information file
    """
    camInfo = getCamInfo(inputDir / 'world.intrinsics')

    # rename some fields, ensure they are numpy arrays
    camInfo['cameraMatrix'] = np.array(camInfo.pop('camera_matrix'))
    camInfo['distCoeff']    = np.array(camInfo.pop('dist_coefs')).flatten()
    camInfo['resolution']   = np.array(camInfo['resolution'])

    # store to file
    storeCameraCalibration(camInfo, outputDir)

    return camInfo['resolution']

def getCameraCalFromBinFile(inputDir: str|pathlib.Path, outputDir: str|pathlib.Path, recInfo: Recording) -> list[int]:
    # provided by pupil-labs
    cal = np.fromfile(
        inputDir / 'calibration.bin',
        np.dtype(
            [
                ("serial", "5a"),
                ("scene_camera_matrix", "(3,3)d"),
                ("scene_distortion_coefficients", "8d"),
                ("scene_extrinsics_affine_matrix", "(3,3)d"),
            ]
        ),
    )
    camInfo = {}
    camInfo['serial_number']= str(cal["serial"])
    camInfo['cameraMatrix'] = cal["scene_camera_matrix"].reshape((3,3))
    camInfo['distCoeff']    = cal["scene_distortion_coefficients"].reshape((8,1))
    camInfo['extrinsic']    = cal["scene_extrinsics_affine_matrix"].reshape((3,3))

    # get resolution from the local intrinsics file or scene video
    camInfo['resolution']   = getSceneCameraResolution(inputDir, recInfo)

    # store to xml file
    storeCameraCalibration(camInfo, outputDir)

    return camInfo['resolution']


def getCameraCalFromOnline(inputDir: str|pathlib.Path, outputDir: str|pathlib.Path, recInfo: Recording) -> list[int]:
    """
    Get camera calibration from pupil labs
    """
    url = f'https://api.cloud.pupil-labs.com/v2/hardware/{recInfo.scene_camera_serial}/calibration.v1?json'

    camInfo = json.loads(urlopen(url).read())
    if camInfo['status'] != 'success':
        raise RuntimeError('Camera calibration could not be loaded, response: %s' % (camInfo['message']))

    camInfo = camInfo['result']

    # rename some fields, ensure they are numpy arrays
    camInfo['cameraMatrix'] = np.array(camInfo.pop('camera_matrix'))
    camInfo['distCoeff']    = np.array(camInfo.pop('dist_coefs')).flatten()
    camInfo['rotation']     = np.reshape(np.array(camInfo.pop('rotation_matrix')),(3,3))

    # get resolution from the local intrinsics file or scene video
    camInfo['resolution']   = getSceneCameraResolution(inputDir, recInfo)

    # store to xml file
    storeCameraCalibration(camInfo, outputDir)

    return camInfo['resolution']

def getCameraCalFromCloudExport(inputDir: str|pathlib.Path, outputDir: str|pathlib.Path, recInfo: Recording) -> list[int]:
    file = inputDir / 'scene_camera.json'
    if not file.is_file():
        return None
    with open(file, 'r') as j:
        camInfo = json.load(j)

    camInfo['cameraMatrix'] = np.array(camInfo.pop('camera_matrix'))
    if 'dist_coefs' in camInfo:
        camInfo['distCoeff']    = np.array(camInfo.pop('dist_coefs')).flatten()
    else:
        camInfo['distCoeff']    = np.array(camInfo.pop('distortion_coefficients')).flatten()

    # get resolution from the scene video
    camInfo['resolution']   = getSceneCameraResolution(inputDir, recInfo)

    # store to xml file
    storeCameraCalibration(camInfo, outputDir)

    return camInfo['resolution']

def storeCameraCalibration(camInfo: dict[str, typing.Any], outputDir: str|pathlib.Path):
    fs = cv2.FileStorage(outputDir / 'calibration.xml', cv2.FILE_STORAGE_WRITE)
    for key,value in camInfo.items():
        fs.write(name=key,val=value)
    fs.release()

def getCamInfo(camInfoFile: str|pathlib.Path):
    with open(camInfoFile, 'rb') as f:
        camInfo = msgpack.unpack(f)

    # get keys which denote a camera resolution
    rex = re.compile('^\(\d+, \d+\)$')

    keys = [k for k in camInfo if rex.match(k)]
    if len(keys)!=1:
        raise RuntimeError('No camera intrinsics or intrinsics for more than one camera found')
    return camInfo[keys[0]]

def getSceneCameraResolution(inputDir: str|pathlib.Path, recInfo: Recording):
    if (inputDir / 'world.intrinsics').is_file():
        return np.array(getCamInfo(inputDir / 'world.intrinsics')['resolution'])
    else:
        import cv2
        cap = cv2.VideoCapture(recInfo.get_scene_video_path())
        if cap.isOpened():
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        return np.array([width, height])


def formatGazeDataPupilPlayer(inputDir: str|pathlib.Path, exportFile: str|pathlib.Path, sceneVideoDimensions: list[int], recInfo: Recording):
    # convert the json file to pandas dataframe
    df = readGazeDataPupilPlayer(exportFile, sceneVideoDimensions, recInfo)

    # get timestamps for the scene video
    frameTs = video_utils.getFrameTimestampsFromVideo(recInfo.get_scene_video_path())

    # check pupil-labs' frame timestamps because we may need to correct
    # frame indices in case of holes in the video
    # also need this to correctly timestamp gaze samples
    if (inputDir / 'world_lookup.npy').is_file():
        ft = pd.DataFrame(np.load(str(inputDir / 'world_lookup.npy')))
        ft['frame_idx'] = ft.index
        ft.loc[ft['container_idx']==-1,'container_frame_idx'] = -1
        needsAdjust = not ft['frame_idx'].equals(ft['container_frame_idx'])
        # prep for later clean up
        toDrop = [x for x in ft.columns if x not in ['frame_idx','timestamp']]
        # do further adjustment that may be needed
        if needsAdjust:
            # not all video frames were encoded into the video file. Need to adjust
            # frame_idx in the gaze data to match actual video file
            temp = pd.merge(df,ft,on='frame_idx')
            temp['frame_idx'] = temp['container_frame_idx']
            temp = temp.rename(columns={'timestamp_x':'timestamp'})
            toDrop.append('timestamp_y')
            df   = temp.drop(columns=toDrop)
    else:
        ft = pd.DataFrame()
        ft['timestamp'] = np.load(str(inputDir / 'world_timestamps.npy'))*1000.0
        ft.index.name = 'frame_idx'
        # check there are no gaps in the video file
        if df['frame_idx'].max() > ft.index.max():
            raise RuntimeError('It appears there are frames missing in the scene video, but the file world_lookup.npy that would be needed to deal with that is missing. You can generate it by opening the recording in pupil player.')

    # set t=0 to video start time
    t0 = ft['timestamp'].iloc[0]*1000-frameTs['timestamp'].iloc[0]
    df.loc[:,'timestamp'] -= t0

    # set timestamps as index for gaze
    df = df.set_index('timestamp')

    return df, frameTs


def readGazeDataPupilPlayer(file: str|pathlib.Path, sceneVideoDimensions: list[int], recInfo: Recording) -> pd.DataFrame:
    """
    convert the gaze_positions.csv file to a pandas dataframe
    """
    isCore = recInfo.eye_tracker is EyeTracker.Pupil_Core

    df = pd.read_csv(file)

    # drop columns with nothing in them
    df = df.dropna(how='all', axis=1)
    df = df.drop(columns=['base_data'],errors='ignore') # drop these columns if they exist)

    # rename and reorder columns
    lookup = {'gaze_timestamp': 'timestamp',
              'world_index': 'frame_idx',
              'eye_center1_3d_x':'gaze_ori_l_x',
              'eye_center1_3d_y':'gaze_ori_l_y',
              'eye_center1_3d_z':'gaze_ori_l_z',
              'gaze_normal1_x':'gaze_dir_l_x',
              'gaze_normal1_y':'gaze_dir_l_y',
              'gaze_normal1_z':'gaze_dir_l_z',
              'eye_center0_3d_x':'gaze_ori_r_x',   # NB: if monocular setup filming left eye, this is the left eye
              'eye_center0_3d_y':'gaze_ori_r_y',
              'eye_center0_3d_z':'gaze_ori_r_z',
              'gaze_normal0_x':'gaze_dir_r_x',
              'gaze_normal0_y':'gaze_dir_r_y',
              'gaze_normal0_z':'gaze_dir_r_z',
              'norm_pos_x':'gaze_pos_vid_x',
              'norm_pos_y':'gaze_pos_vid_y',
              'gaze_point_3d_x': 'gaze_pos_3d_x',
              'gaze_point_3d_y': 'gaze_pos_3d_y',
              'gaze_point_3d_z': 'gaze_pos_3d_z'}
    df=df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    idx.extend([x for x in df.columns if x not in idx])   # append columns not in lookup
    df = df[idx]

    # mark data with insufficient confidence as missing.
    # for pupil core, pupil labs recommends a threshold of 0.6,
    # for the pupil invisible its a binary signal, and
    # confidence 0 should be excluded
    confThresh = 0.6 if isCore else 0
    todo = [x for x in idx if x in lookup.values()]
    toRemove = df.confidence <= confThresh
    for c in todo[2:]:
        df.loc[toRemove,c] = np.nan

    # convert timestamps from s to ms
    df.loc[:,'timestamp'] *= 1000.0

    # turn gaze locations into pixel data with origin in top-left
    df.loc[:,'gaze_pos_vid_x'] *= sceneVideoDimensions[0]
    df.loc[:,'gaze_pos_vid_y'] = (1-df.loc[:,'gaze_pos_vid_y'])*sceneVideoDimensions[1] # turn origin from bottom-left to top-left

    # return the dataframe
    return df

def formatGazeDataCloudExport(inputDir: str|pathlib.Path, exportFile: str|pathlib.Path):
    df = readGazeDataCloudExport(exportFile)

    frameTimestamps = pd.read_csv(inputDir/'world_timestamps.csv')
    frameTimestamps = frameTimestamps.rename(columns={'timestamp [ns]': 'timestamp'})
    frameTimestamps = frameTimestamps.drop(columns=[x for x in frameTimestamps.columns if x not in ['timestamp']])
    frameTimestamps['frame_idx'] = frameTimestamps.index
    frameTimestamps = frameTimestamps.set_index('frame_idx')

    # set t=0 to video start time
    t0_ns = frameTimestamps['timestamp'].iloc[0]
    df.loc[:,'timestamp']               -= t0_ns
    frameTimestamps.loc[:,'timestamp']  -= t0_ns
    df.loc[:,'timestamp']               /= 1000000.0    # convert timestamps from ns to ms
    frameTimestamps.loc[:,'timestamp']  /= 1000000.0

    # set timestamps as index for gaze
    df = df.set_index('timestamp')

    # use the frame timestamps to assign a frame number to each data point
    frameIdx = video_utils.tssToFrameNumber(df.index,frameTimestamps['timestamp'].to_numpy())
    df.insert(0,'frame_idx',frameIdx['frame_idx'])

    return df, frameTimestamps


def readGazeDataCloudExport(file: str|pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(file)

    # rename and reorder columns
    lookup = {'timestamp [ns]': 'timestamp',
              'gaze x [px]': 'gaze_pos_vid_x',
              'gaze y [px]': 'gaze_pos_vid_y',
              'pupil diameter left [mm]': 'pup_diam_l',
              'pupil diameter right [mm]': 'pup_diam_r',
              'eyeball center left x [mm]': 'gaze_ori_l_x',
              'eyeball center left y [mm]': 'gaze_ori_l_y',
              'eyeball center left z [mm]': 'gaze_ori_l_z',
              'eyeball center right x [mm]': 'gaze_ori_r_x',
              'eyeball center right y [mm]': 'gaze_ori_r_y',
              'eyeball center right z [mm]': 'gaze_ori_r_z',
              'optical axis left x': 'gaze_dir_l_x',
              'optical axis left y': 'gaze_dir_l_y',
              'optical axis left z': 'gaze_dir_l_z',
              'optical axis right x': 'gaze_dir_r_x',
              'optical axis right y': 'gaze_dir_r_y',
              'optical axis right z': 'gaze_dir_r_z',
    }
    df=df.drop(columns=[x for x in df.columns if x not in lookup and x not in ['worn','blink id']])
    df=df.rename(columns=lookup)

    # check if there is an eye states file
    eye_state_file = file.parent / '3d_eye_states.csv'
    if eye_state_file.exists():
        df_eye = pd.read_csv(eye_state_file)
        df_eye = df_eye.drop(columns=[x for x in df_eye.columns if x not in lookup])
        df_eye = df_eye.rename(columns=lookup)
        df = df.join(df_eye.set_index('timestamp'), on='timestamp')

    # mark data where eye tracker is not worn or during blink as missing
    todo = [lookup[k] for k in lookup if lookup[k] in df.columns and lookup[k]!='timestamp']
    toRemove = df.worn == 0
    if 'blink id' in df:
        toRemove = np.logical_or(toRemove, df['blink id']>0)
    for c in todo:
        df.loc[toRemove,c] = np.nan

    # remove last columns we don't need anymore
    df=df.drop(columns=[x for x in df.columns if x in ['worn','blink id']])

    return df