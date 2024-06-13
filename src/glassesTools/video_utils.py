import numpy as np
import pandas as pd
import cv2

from .mp4analyser import iso


def av_rescale(a, b, c):
    # a * b / c, rounding to nearest and halfway cases away from zero
    # e.g., scale a from timebase c to timebase b
    # from ffmpeg libavutil mathematics.c, porting the simple case assuming that a, b and c <= INT_MAX
    r = c // 2
    return (a * b + r) // c


def getFrameTimestampsFromVideo(vid_file):
    """
    Parse the supplied video, return an array of frame timestamps. There must be only one video stream
    in the video file, because otherwise we do not know which is the correct stream.
    """
    if vid_file.suffix in ['.mov', '.mp4', '.m4a', '.3gp', '.3g2', '.mj2']:
        # parse mp4 file
        boxes       = iso.Mp4File(str(vid_file))
        summary     = boxes.get_summary()
        vid_tracks  = [t for t in summary['track_list'] if t['media_type']=='video']
        assert len(vid_tracks)==1, f"File has {len(vid_tracks)} video tracks (more than one), not supported"
        # 1. find mdat box
        moov        = boxes.children[[i for i,x in enumerate(boxes.children) if x.type=='moov'][0]]
        # 2. get global/movie time scale
        movie_time_scale = np.int64(moov.children[[i for i,x in enumerate(moov.children) if x.type=='mvhd'][0]].box_info['timescale'])
        # 3. find video track boxes
        trak_idxs   = [i for i,x in enumerate(moov.children) if x.type=='trak']
        trak_idxs   = [x for i,x in enumerate(trak_idxs) if summary['track_list'][i]['media_type']=='video']
        assert len(trak_idxs)==1
        trak        = moov.children[trak_idxs[0]]
        # 4. get mdia box
        mdia        = trak.children[[i for i,x in enumerate(trak.children) if x.type=='mdia'][0]]
        # 5. get media/track time_scale and duration fields from mdhd
        mdhd            = mdia.children[[i for i,x in enumerate(mdia.children) if x.type=='mdhd'][0]]
        media_time_scale= mdhd.box_info['timescale']
        # 6. check for presence of edit list
        # if its there, check its one we understand (one or multiple empty list at the beginning
        # to shift movie start, and/or a single non-empty edit list), and parse it. Else abort
        edts_idx= [i for i,x in enumerate(trak.children) if x.type=='edts']
        empty_duration  = np.int64(0)
        media_start_time= np.int64(-1)
        media_duration  = np.int64(-1)
        if edts_idx:
            elst = trak.children[edts_idx[0]].children[0]
            edit_start_index = 0
            # logic ported from mov_build_index()/mov_fix_index() in ffmpeg's libavformat/mov.c
            for i in range(elst.box_info['entry_count']):
                if i==edit_start_index and elst.box_info['entry_list'][i]['media_time'] == -1:
                    # empty edit list, indicates the start time of the stream
                    # relative to the presentation itself
                    this_empty_duration  = np.int64(elst.box_info['entry_list'][i]['segment_duration'])  # NB: in movie time scale
                    # convert duration from edit list from global timescale to track timescale
                    empty_duration  += av_rescale(this_empty_duration,media_time_scale,movie_time_scale)
                    edit_start_index+= 1
                elif i==edit_start_index and elst.box_info['entry_list'][i]['media_time'] > 0:
                    media_start_time = np.int64(elst.box_info['entry_list'][i]['media_time'])   # NB: already in track timescale, do not scale
                    media_duration   = av_rescale(np.int64(elst.box_info['entry_list'][i]['segment_duration']),media_time_scale,movie_time_scale)   # as above, scale to track timescale
                    if media_start_time<0:
                        raise RuntimeError('File contains an edit list that is too complicated (media start time < 0) for this parser, not supported')
                    if elst.box_info['entry_list'][i]['media_rate']!=1.0:
                        raise RuntimeError('File contains an edit list that is too complicated (media time is not 1.0) for this parser, not supported')
                elif i>edit_start_index:
                    raise RuntimeError('File contains an edit list that is too complicated (multiple non-empty edits) for this parser, not supported')
        # 7. get stbl
        minf    = mdia.children[[i for i,x in enumerate(mdia.children) if x.type=='minf'][0]]
        stbl    = minf.children[[i for i,x in enumerate(minf.children) if x.type=='stbl'][0]]
        # 8. check whether we have a ctts atom
        ctts_idx= [i for i,x in enumerate(stbl.children) if x.type=='ctts']
        if ctts_idx:
            ctts_ = stbl.children[ctts_idx[0]].box_info['entry_list']
            if any([e['sample_offset']<0 for e in ctts_]):
                # if we need to deal with them, we'd probably want to completely port mov_build_index() and mov_fix_index() in ffmpeg's libavformat/mov.c
                raise RuntimeError('Encountered a ctts (composition offset) atom with negative sample offsets, cannot handle that. aborting.')
            # uncompress
            total_frames_ctts = sum([e['sample_count'] for e in ctts_])
            ctts = np.zeros(total_frames_ctts, dtype=np.int64)
            idx = 0
            for e in ctts_:
                ctts[idx:idx+e['sample_count']] = e['sample_offset']
                idx = idx+e['sample_count']
        # 9. get sample table from stts
        stts = stbl.children[[i for i,x in enumerate(stbl.children) if x.type=='stts'][0]].box_info['entry_list']
        # uncompress the delta table
        total_frames_stts = sum([e['sample_count'] for e in stts])
        dts = np.zeros(total_frames_stts, dtype=np.int64)
        idx = 0
        for e in stts:
            dts[idx:idx+e['sample_count']] = e['sample_delta']
            idx = idx+e['sample_count']

        # 10. now put it all together
        # turn into timestamps
        dts = np.cumsum(np.insert(dts, 0, 0))
        dts = np.delete(dts,-1) # remove last, since that denotes _end_ of last frame, and we only need timestamps for frame onsets
        # apply ctts
        if ctts_idx:
            dts += ctts
            # ctts should lead to a reordering of frames, so sort
            dts = np.sort(dts)
        # if we have a non-empty edit list, apply
        if media_start_time!=-1:
            # remove all timestamps before start or after end of edit list
            to_keep = np.logical_or(dts >= media_start_time, dts < (media_duration + media_start_time))
            dts = dts[to_keep]
            min_corrected_pts = dts[0]  # already sorted, this is the first frame's pts
            # If there are empty edits, then min_corrected_pts might be positive
            # intentionally. So we subtract the sum duration of emtpy edits here.
            min_corrected_pts -= empty_duration
            # If the minimum pts turns out to be greater than zero,
            # then we subtract the dts by that amount to make the first pts zero.
            dts -= min_corrected_pts
        # now turn into timestamps in ms
        frameTs = (dts+empty_duration)/media_time_scale*1000
    else:
        # open file with opencv and get timestamps of each frame
        vid = cv2.VideoCapture(vid_file)
        nframes = float(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frameTs = []
        frame_idx = 0
        while vid.isOpened():
            # read timestamp _before_ reading the frame, so we get position at start of the frame, not at
            # end
            ts = vid.get(cv2.CAP_PROP_POS_MSEC)

            ret, _ = vid.read()
            frame_idx += 1

            # You'd think to get the time at the start of the frame, which is what we want, you'd need to
            # read the time _before_ reading the frame. But there seems to be an off-by-one here for some
            # files, like at least some MP4s, but not in some AVIs in my testing. Catch this off-by-one
            # and to correct for it, do not store the first timestamp. This ensures we get a sensible
            # output (else first two frames have timestamp 0.0 ms).
            if frame_idx==1 and ts==vid.get(cv2.CAP_PROP_POS_MSEC):
                continue

            frameTs.append(ts)
            # check if we're done. Can't trust ret==False to indicate we're at end of video, as it may also return false for some frames when video has errors in the middle that we can just read past
            if (not ret and frame_idx>0 and frame_idx/nframes<.99):
                raise RuntimeError("The video file is corrupt. Testing has shown that it cannot be guaranteed that timestamps remain correct when trying to read past the hole. So abort, cannot process this video.")

        # release the video capture object
        vid.release()
        frameTs = np.array(frameTs)

    ### convert the frame_timestamps to dataframe
    frameIdx = np.arange(0, len(frameTs))
    frameTsDf = pd.DataFrame({'frame_idx': frameIdx, 'timestamp': frameTs})
    frameTsDf.set_index('frame_idx', inplace=True)

    return frameTsDf


def tssToFrameNumber(ts,frameTimestamps,mode='nearest',trim=False):
    df = pd.DataFrame(index=ts)
    df.insert(0,'frame_idx',np.int64(0))
    if isinstance(frameTimestamps, list):
        frameTimestamps = np.array(frameTimestamps)

    # get index where this ts would be inserted into the frame_timestamp array
    idxs = np.searchsorted(frameTimestamps, ts)
    if mode=='after':
        idxs = idxs.astype('float32')
        # out of range, set to nan
        idxs[idxs==0] = np.nan
        # -1: since idx points to frame timestamp for frame after the one during which the ts ocurred, correct
        idxs -= 1
    elif mode=='nearest':
        # implementation from https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python/8929827#8929827
        # same logic as used by pupil labs
        idxs = np.clip(idxs, 1, len(frameTimestamps)-1)
        left = frameTimestamps[idxs-1]
        right = frameTimestamps[idxs]
        idxs -= ts - left < right - ts

    if trim:
        # set any timestamps before to -1
        idxs[df.index<0] = -1
        # get average frame interval, and set data beyond framets+1 extra frame to -1 as well
        avIFI = np.mean(np.diff(df.index.to_numpy()))
        idxs[df.index>frameTimestamps[-1]+avIFI] = -1

    df=df.assign(frame_idx=idxs)
    if mode=='after':
        df=df.convert_dtypes() # turn into int64 again

    return df