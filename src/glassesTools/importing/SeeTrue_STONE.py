"""
Cast raw SeeTrue data into common format.

The output directory will contain:
    - frameTimestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gazeData.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera
"""

import shutil
import pathlib
import os
import cv2
import pandas as pd
import numpy as np

from ..recording import Recording
from ..eyetracker import EyeTracker
from .. import video_utils


def preprocessData(output_dir: str|pathlib.Path, source_dir: str|pathlib.Path=None, rec_info: Recording=None, cam_cal_file: str|pathlib.Path=None, copy_scene_video = True) -> Recording:
    from . import check_folders, _store_data
    # NB: copy_scene_video input argument is ignored, SeeTrue recordings must be transcoded with ffmpeg to be useful

    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found on path. ffmpeg is required for importing SeeTrue recordings. Cannot continue')

    """
    Run all preprocessing steps on SeeTrue STONE data and store in output_dir
    """
    output_dir, source_dir, rec_info = check_folders(output_dir, source_dir, rec_info, EyeTracker.SeeTrue_STONE)
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
        shutil.copyfile(str(cam_cal_file), str(output_dir / 'calibration.xml'))
    else:
        print('    !! No camera calibration provided! Defaulting to hardcoded')
        getCameraHardcoded(output_dir)


    _store_data(output_dir, gazeDf, frameTimestamps, rec_info)

    return rec_info


def getRecordingInfo(inputDir: str|pathlib.Path) -> list[Recording]:
    # returns None if not a recording directory
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
    framerate = "{:.4f}".format(1000./ifi)
    outFile = outputDir / 'worldCamera.mp4'
    cmd_str = ' '.join(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2', '-framerate', framerate, '-start_number', str(frames[0]), '-i', '"'+str(sceneVidDir / 'frame_%d.jpeg')+'"', '"'+str(outFile)+'"'])
    os.system(cmd_str)
    if outFile.is_file():
        recInfo.scene_video_file = outFile.name
    else:
        raise RuntimeError('Error making a scene video out of the SeeTrue''s frames')

    # attempt 2 that should allow correct VFR video files, but doesn't work with current MediaWriter
    # due to what i think is a bug: https://github.com/matham/ffpyplayer/issues/129.
    ## get which pixel format
    #codec    = ffpyplayer.tools.get_format_codec(fmt='mp4')
    #pix_fmt  = ffpyplayer.tools.get_best_pix_fmt('bgr24',ffpyplayer.tools.get_supported_pixfmts(codec))
    #fpsFrac  = Fraction(1000./ifi).limit_denominator(10000).as_integer_ratio()
    #fpsFrac  = tuple([x*10 for x in fpsFrac])
    ## scene video
    #out_opts = {'pix_fmt_in':'bgr24', 'pix_fmt_out':pix_fmt, 'width_in':w, 'height_in':h,'frame_rate':fpsFrac}
    #vidOut   = MediaWriter(str(outputDir / 'worldCamera.mp4'), [out_opts], overwrite=True)
    #t0       = frameTimestamps.index[0]
    #for i,f in enumerate(frames):
    #    frame = cv2.imread(str(sceneVidDir / 'frame_{:5d}.jpeg'.format(f)))
    #    img   = Image(plane_buffers=[frame.flatten().tobytes()], pix_fmt='bgr24', size=(int(w), int(h)))
    #    t = (frameTimestamps.index[i]-t0)/1000
    #    print(t, t/(fpsFrac[1]/fpsFrac[0]))
    #    vidOut.write_frame(img=img, pts=t)

    # delete the black frames we added, if any
    for f in blackFrames:
        if (sceneVidDir / 'frame_{:d}.jpeg'.format(f)).is_file():
            (sceneVidDir / 'frame_{:d}.jpeg'.format(f)).unlink(missing_ok=True)

    # 4. write data to file
    # fix up frame idxs and timestamps
    firstFrame = frameTimestamps['frame_idx'].min()

    # write the gaze data to a csv file
    gazeDf['frame_idx'] -= firstFrame

    # also store frame timestamps
    # this is what it should be if we get VFR video file writing above to work
    #frameTimestamps['frame_idx'] -= firstFrame
    #frameTimestamps=frameTimestamps.reset_index().set_index('frame_idx')
    #frameTimestamps['timestamp'] -= frameTimestamps['timestamp'].min()
    # instead now, get actual ts for each frame in written video as that is what we
    # have to work with. Note that these do not match gaze data ts, but code nowhere
    # assumes they do
    frameTimestamps = video_utils.getFrameTimestampsFromVideo(recInfo.get_scene_video_path())

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
    fs = cv2.FileStorage(outputDir / 'calibration.xml', cv2.FILE_STORAGE_WRITE)
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

    # remove unneeded columns
    rem = [x for x in df.columns if x not in ['Frame number','Timestamp','Gazepoint X','Gazepoint Y','Scene picture number']]
    df = df.drop(columns=rem)

    # rename and reorder columns
    lookup = {'Timestamp': 'timestamp',
              'Scene picture number': 'frame_idx',
              'Gazepoint X': 'gaze_pos_vid_x',
              'Gazepoint Y': 'gaze_pos_vid_y',}
    df=df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    idx.extend([x for x in df.columns if x not in idx])   # append columns not in lookup
    df = df[idx]

    # set timestamps as index
    df = df.set_index('timestamp')

    # turn gaze locations into pixel data with origin in top-left
    df['gaze_pos_vid_x'] *= sceneVideoDimensions[0]
    df['gaze_pos_vid_y'] *= sceneVideoDimensions[1]

    # return the dataframe
    return df