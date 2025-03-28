import shutil
import pathlib
import json
import pandas as pd

from ..recording import Recording
from ..eyetracker import EyeTracker
from .. import naming, timestamps, video_utils

def importData(output_dir: str|pathlib.Path=None, source_dir: str|pathlib.Path=None, rec_info: Recording=None, copy_scene_video = True, source_dir_as_relative_path = False) -> Recording:
    from . import check_folders, _store_data
    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, EyeTracker.Meta_Aria_Gen_1)
    print(f'processing: {source_dir.name} -> {output_dir}')

    ### check and copy needed files to the output directory
    print('  Check and copy raw data...')
    ### check recording and get export directory
    if rec_info is not None:
        checkRecording(source_dir, rec_info)
    else:
        rec_info = getRecordingInfo(source_dir)
        if rec_info is None:
            raise RuntimeError(f"The folder {source_dir} is not recognized as a {EyeTracker.Meta_Aria_Gen_1.value} recording.")

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()


    ### copy the raw data to the output directory
    srcVid, destVid, gazeData, frameTimestamps = copyRecording(source_dir, output_dir, rec_info, copy_scene_video)
    if destVid:
        rec_info.scene_video_file = destVid.name
    else:
        rec_info.scene_video_file =  srcVid.name

    _store_data(output_dir, gazeData, frameTimestamps, rec_info, source_dir_as_relative_path=source_dir_as_relative_path)

    return rec_info

def getRecordingInfo(inputDir: str|pathlib.Path) -> Recording:
    # returns None if not a recording directory
    inputDir = pathlib.Path(inputDir)
    recInfo = Recording(source_directory=inputDir, eye_tracker=EyeTracker.Meta_Aria_Gen_1)

    # check expected files are present
    for f in ('metadata.json','worldCamera.mp4','gaze.tsv','calibration.xml'):
        if not (inputDir/f).is_file():
            print(f'This directory does not contain a valid {EyeTracker.Meta_Aria_Gen_1.value} recording export. The {f} file is not found in the input directory {inputDir}. Make sure you run the meta_aria_gen1_exporter.py script on the recording\'s vrs file')
            return None

    with open(inputDir/'metadata.json', 'r') as f:
        metadata = json.load(f)
    recInfo.name = metadata['name']
    recInfo.scene_camera_serial = metadata['scene_camera_serial']
    recInfo.duration = int(metadata['duration']/1000)   # in us, convert to ms
    recInfo.glasses_serial = metadata['glasses_serial']
    recInfo.start_time = timestamps.Timestamp(metadata['start_time'])
    recInfo.scene_video_file = 'worldCamera.mp4'

    # we got a valid recording
    # return what we've got
    return recInfo

def checkRecording(inputDir: str|pathlib.Path, recInfo: Recording):
    actualRecInfo = getRecordingInfo(inputDir)

    if actualRecInfo is None or recInfo.name!=actualRecInfo.name:
        raise ValueError(f"A recording with the name \"{recInfo.name}\" was not found in the folder {inputDir}.")

    # make sure caller did not mess with recInfo
    if recInfo.eye_tracker!=actualRecInfo.eye_tracker:
        raise ValueError(f"A recording for a \"{recInfo.eye_tracker.value}\" device was not found in the folder {inputDir}.")

def copyRecording(inputDir: pathlib.Path, outputDir: pathlib.Path, rec_info: Recording, copy_scene_video:bool):
    gazeFile = inputDir/'gaze.tsv'
    if not gazeFile.is_file():
        raise RuntimeError(f'The {gazeFile} file is not found in the input directory {inputDir}')
    gazeData = pd.read_csv(gazeFile,sep='\t',index_col='timestamp')

    vidSrcFile = inputDir/'worldCamera.mp4'
    if not vidSrcFile.is_file():
        raise RuntimeError(f'The {vidSrcFile} file is not found in the input directory {inputDir}')
    if copy_scene_video:
        vidDestFile = outputDir / f'{naming.scene_camera_video_fname_stem}.mp4'
        shutil.copy2(vidSrcFile, vidDestFile)
    else:
        vidDestFile = None

    # get video timestamps
    frameTimestamps = video_utils.get_frame_timestamps_from_video(rec_info.get_scene_video_path())

    # convert gaze timestamps from us to ms
    gazeData = gazeData.reset_index()
    gazeData.loc[:,'timestamp'] /= 1000.0
    # add frame_idx for each gaze sample
    frameIdx = video_utils.timestamps_to_frame_number(gazeData.loc[:,'timestamp'].values,frameTimestamps['timestamp'].values)
    gazeData.insert(1,'frame_idx',frameIdx['frame_idx'].values)
    gazeData = gazeData.set_index('timestamp')

    cal_file = inputDir/'calibration.xml'
    if not cal_file.is_file():
        raise RuntimeError(f'The {cal_file} file is not found in the input directory {inputDir}')
    shutil.copy2(cal_file, outputDir / naming.scene_camera_calibration_fname)

    return vidSrcFile, vidDestFile, gazeData, frameTimestamps