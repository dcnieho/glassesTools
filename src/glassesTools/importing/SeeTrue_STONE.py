"""
Cast raw SeeTrue data into common format.

The output directory will contain:
    - frameTimestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gazeData.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera
"""

import shutil
import pathlib
import cv2
import pandas as pd
import numpy as np
import av
from fractions import Fraction

from ..recording import Recording
from ..eyetracker import EyeTracker
from .. import naming


def preprocessData(output_dir: str|pathlib.Path, source_dir: str|pathlib.Path=None, rec_info: Recording=None, copy_scene_video = True, source_dir_as_relative_path = False, cam_cal_file: str|pathlib.Path=None) -> Recording:
    from . import check_folders, _store_data
    # NB: copy_scene_video input argument is ignored, SeeTrue recordings must be transcoded with ffmpeg to be useful

    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found on path. ffmpeg is required for importing SeeTrue recordings. Cannot continue')

    """
    Run all preprocessing steps on SeeTrue STONE data and store in output_dir
    """
    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, EyeTracker.SeeTrue_STONE)
    print(f'processing: {source_dir.name} -> {output_dir}')


    ### check and copy needed files to the output directory
    print('  Check and copy raw data...')
    if rec_info is not None:
        if not checkRecording(source_dir, rec_info):
            raise ValueError(f"A recording with the name \"{rec_info.name}\" was not found in the folder {source_dir}.")
    else:
        recInfos = getRecordingInfo(source_dir)
        if recInfos is None:
            raise RuntimeError(f"The folder {source_dir} does not contain SeeTrue STONE recordings.")
        rec_info = recInfos[0]  # take first, arbitrarily. If anything else wanted, user should call this function with a correct rec_info themselves

    # make output dirs
    if not output_dir.is_dir():
        output_dir.mkdir()


    #### prep the data
    # NB: gaze data and scene video prep are intertwined, status messages are output inside this function
    gazeDf, frameTimestamps = copySeeTrueRecording(source_dir, output_dir, rec_info)

    print('  Getting camera calibration...')
    if cam_cal_file is not None:
        shutil.copyfile(str(cam_cal_file), str(output_dir / naming.scene_camera_calibration_fname))
    else:
        print('    !! No camera calibration provided! Defaulting to hardcoded')
        getCameraHardcoded(output_dir)


    _store_data(output_dir, gazeDf, frameTimestamps, rec_info, source_dir_as_relative_path=source_dir_as_relative_path)

    return rec_info


def getRecordingInfo(inputDir: str|pathlib.Path) -> list[Recording]:
    # returns None if not a recording directory
    inputDir = pathlib.Path(inputDir)
    recInfos = []

    # NB: a SeeTrue directory may contain multiple recordings

    # get recordings. These are indicated by the sequence number in both EyeData.csv and ScenePics folder names
    for r in inputDir.glob('*.csv'):
        if not str(r.name).startswith('EyeData'):
            # print(f"file {r.name} not recognized as a recording (wrong name, should start with 'EyeData'), skipping")
            continue

        # get sequence number
        _,recording = r.stem.split('_')

        # check there is a matching scenevideo
        sceneVidDir = r.parent / ('ScenePics_' + recording)
        if not sceneVidDir.is_dir():
            # print(f"folder {sceneVidDir} not found, meaning there is no scene video for this recording, skipping")
            continue

        recInfos.append(Recording(source_directory=inputDir, eye_tracker=EyeTracker.SeeTrue_STONE))
        recInfos[-1].participant = inputDir.name
        recInfos[-1].name = recording

    # should return None if no valid recordings found
    return recInfos if recInfos else None


def checkRecording(inputDir: str|pathlib.Path, recInfo: Recording):
    """
    This checks that the folder is properly prepared
    (i.e. the required BeGaze exports were run)
    """
    # check we have an exported gaze data file
    file = f'EyeData_{recInfo.name}.csv'
    if not (inputDir / file).is_file():
        return False

    # check we have an exported scene video
    file = f'ScenePics_{recInfo.name}'
    if not (inputDir / file).is_dir():
        return False

    return True


def copySeeTrueRecording(inputDir: pathlib.Path, outputDir: pathlib.Path, recInfo: Recording):
    """
    Copy the relevant files from the specified input dir to the specified output dirs
    """

    # get scene video dimensions by interrogating a frame in sceneVidDir
    sceneVidDir = inputDir / ('ScenePics_' + recInfo.name)
    frame = next(sceneVidDir.glob('*.jpeg'))
    h,w,_ = cv2.imread(frame).shape

    # prep gaze data and get video frame timestamps from it
    print('  Prepping gaze data...')
    file = f'EyeData_{recInfo.name}.csv'
    gazeDf, frameTimestamps = formatGazeData(inputDir / file, [w,h])

    # make scene video
    print('  Prepping scene video...')
    # 1. see if there are frames missing, insert black frames if so
    frames = []
    for f in sceneVidDir.glob('*.jpeg'):
        _,fr = f.stem.split('_')
        frames.append(int(fr))
    frames = sorted(frames)

    # 2. see if framenumbers are as expected from the gaze data file
    # get average ifi
    ifi = np.mean(np.diff(frameTimestamps.index))
    # 2.1 remove frame timestamps that are before the first frame for which we have an image
    frameTimestamps=frameTimestamps.drop(frameTimestamps[frameTimestamps.frame_idx < frames[ 0]].index)
    # 2.2 remove frame timestamps that are beyond last frame for which we have an image
    frameTimestamps=frameTimestamps.drop(frameTimestamps[frameTimestamps.frame_idx > frames[-1]].index)
    # 2.3 add frame timestamps for images we have before first eye data
    if frames[ 0] < frameTimestamps.iloc[ 0,:].to_numpy()[0]:
        nFrames = frameTimestamps.iloc[ 0,:].to_numpy()[0] - frames[ 0]
        t0 = frameTimestamps.index[0]
        f0 = frameTimestamps.iloc[ 0,:].to_numpy()[0]
        for f in range(-1,-(nFrames+1),-1):
            frameTimestamps.loc[t0+f*ifi] = f0+f
        frameTimestamps = frameTimestamps.sort_index()
    # 2.4 add frame timestamps for images we have after last eye data
    if frames[-1] > frameTimestamps.iloc[-1,:].to_numpy()[0]:
        nFrames = frames[-1] - frameTimestamps.iloc[-1,:].to_numpy()[0]
        t0 = frameTimestamps.index[-1]
        f0 = frameTimestamps.iloc[-1,:].to_numpy()[0]
        for f in range(1,nFrames+1):
            frameTimestamps.loc[t0+f*ifi] = f0+f
        frameTimestamps = frameTimestamps.sort_index()
    # 2.5 check if holes, fill
    blackFrames = []
    frameDelta = np.diff(frames)
    if np.any(frameDelta>1):
        # frames images missing, add them (NB: if timestamps also missing, thats dealt with below)
        idxGaps = np.argwhere(frameDelta>1).flatten()     # idxGaps is last idx before each gap
        frGaps  = np.array(frames)[idxGaps].flatten()
        nFrames = frameDelta[idxGaps].flatten()
        for b,x in zip(frGaps+1,nFrames):
            for y in range(x-1):
                blackFrames.append(b+y)

        # make black frame
        blackIm = np.zeros((h,w,3), np.uint8)   # black image
        for f in blackFrames:
            # store black frame to file
            cv2.imwrite(sceneVidDir / f'frame_{f:d}.jpeg',blackIm)
            frames.append(f)
        frames = sorted(frames)

    # 3. make into video
    # 3.1 find unique scene camera frames and their timestamps
    firstFrame      = frameTimestamps['frame_idx'].min()
    firstFrameTs    = frameTimestamps['frame_idx'].idxmin()
    frameTimestamps = frameTimestamps.reset_index().groupby('frame_idx').first().reset_index()
    # 3.2 make concat filter input file
    concat_file = outputDir/'concat_input.txt'
    with open(concat_file, 'wt') as f:
        f.writelines('ffconcat version 1.0\n')
        fnames = (f'frame_{f}.jpeg' for f in frameTimestamps['frame_idx'].to_numpy())
        f.writelines((f"file '{sceneVidDir / fn}'\n" for fn in fnames))

    # 3.3 determine frame pts and durations
    ifis            = np.diff(frameTimestamps['timestamp'].to_numpy())
    ifis            = np.append(ifis,[np.median(ifis)])
    durs            = ifis/1000
    pts_time        = np.cumsum(np.append([0],durs))

    # 3.4 read frames through concat filter, output to mp4 with the right pts and dur
    outFile = outputDir / f'{naming.scene_camera_video_fname_stem}.mp4'
    ts = 900000
    with av.open(concat_file, 'r', format='concat', options={'safe':'0'}) as inp:
        in_stream = inp.streams.video[0]
        with av.open(outFile, 'w', format='mp4') as out:
            out_stream = out.add_stream('libx264')
            out_stream.width = in_stream.codec_context.width  # Set frame width to be the same as the width of the input stream
            out_stream.height = in_stream.codec_context.height  # Set frame height to be the same as the height of the input stream
            out_stream.pix_fmt = in_stream.codec_context.pix_fmt  # Copy pixel format from input stream to output stream
            out_stream.time_base = Fraction(1, ts)

            for frame_idx, frame in enumerate(inp.decode(in_stream)):
                frame.pts       = np.round(pts_time[frame_idx]/out_stream.time_base)
                frame.dts       = np.round(pts_time[frame_idx]/out_stream.time_base)
                frame.duration  = np.round(  durs  [frame_idx]/out_stream.time_base)
                frame.time_base = out_stream.time_base

                packet = out_stream.encode(frame)
                out.mux(packet)
            
            # Flush the encoder
            packet = out_stream.encode(None)
            out.mux(packet)

    # 3.5 clean up
    # check for success
    concat_file.unlink(missing_ok=True)
    if outFile.is_file():
        recInfo.scene_video_file = outFile.name
    else:
        raise RuntimeError('Error making a scene video out of the SeeTrue''s frames')

    # delete the black frames we added, if any
    for f in blackFrames:
        if (sceneVidDir / 'frame_{:d}.jpeg'.format(f)).is_file():
            (sceneVidDir / 'frame_{:d}.jpeg'.format(f)).unlink(missing_ok=True)

    # 4. fix up frame idxs and timestamps in gaze and video data
    # prep the gaze timestamps
    gazeDf.index -= firstFrameTs
    # overwrite the video frames, now we have one video frame per gaze sample
    gazeDf['frame_idx'] -= firstFrame

    # Also fix video timestamps
    frameTimestamps = frameTimestamps.drop(columns=['frame_idx'])
    frameTimestamps.index.name = 'frame_idx'
    frameTimestamps['timestamp'] -= firstFrameTs

    return gazeDf, frameTimestamps


def getCameraHardcoded(outputDir: str|pathlib.Path):
    """
    Get camera calibration
    Hardcoded as per info received from SeeTrue
    """
    # turn into camera matrix and distortion coefficients as used by OpenCV
    camera = {}
    camera['cameraMatrix'] = np.array([[495,   0, 300],
                                       [  0, 495, 255],
                                       [  0,   0,   1]], dtype=np.float64)
    camera['distCoeff'] = np.array([-0.55, 0.4, 0, 0, -0.2])
    camera['resolution'] = np.array([640, 480])

    # store to file
    fs = cv2.FileStorage(outputDir / naming.scene_camera_calibration_fname, cv2.FILE_STORAGE_WRITE)
    for key,value in camera.items():
        fs.write(name=key,val=value)
    fs.release()


def formatGazeData(inputFile: str|pathlib.Path, sceneVideoDimensions: list[int]):
    """
    load gazedata file
    format to get the gaze coordinates w.r.t. world camera, and timestamps for
    every frame of video

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps
    """

    # convert the json file to pandas dataframe
    df = gazedata2df(inputFile, sceneVideoDimensions)

    # get time stamps for scene picture numbers
    frameTimestamps = pd.DataFrame(df['frame_idx'])

    # return the gaze data df and frame time stamps array
    return df, frameTimestamps


def gazedata2df(textFile: str|pathlib.Path, sceneVideoDimensions: list[int]):
    """
    convert the gazedata file to a pandas dataframe
    """

    df = pd.read_table(textFile,sep=';',index_col=False)
    df.columns=df.columns.str.strip()

    # rename and reorder columns
    lookup = {'Timestamp': 'timestamp',
              'Scene picture number': 'frame_idx',
              'Gazepoint X': 'gaze_pos_vid_x',
              'Gazepoint Y': 'gaze_pos_vid_y',
              'Pupil area (left), sq mm' : 'pup_diam_l',
              'Pupil area (right), sq mm': 'pup_diam_r'}
    df=df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    df = df[idx]

    # pupil area to diameter
    df['pup_diam_l' ] = 2*np.sqrt(df['pup_diam_l'].to_numpy()/np.pi)
    df['pup_diam_r']  = 2*np.sqrt(df['pup_diam_r'].to_numpy()/np.pi)

    # set timestamps as index
    df = df.set_index('timestamp')

    # turn gaze locations into pixel data with origin in top-left
    df['gaze_pos_vid_x'] *= sceneVideoDimensions[0]
    df['gaze_pos_vid_y'] *= sceneVideoDimensions[1]

    # return the dataframe
    return df