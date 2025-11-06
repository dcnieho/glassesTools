import pathlib
import numpy as np
import pandas as pd
import math
import warnings

from .. import data_types, gaze_worldref, naming, pose, transforms


def compute(
        gazes: str|pathlib.Path|dict[int, list[gaze_worldref.Gaze]],
        poses: str|pathlib.Path|dict[int, pose.Pose],
        marker_intervals: str|pathlib.Path|pd.DataFrame,
        validation_intervals: list[list[int]],
        targets: dict[int, np.ndarray],
        distance_mm_for_homography: float,
        output_directory: str|pathlib.Path,
        filename: str = naming.validation_offset_fname,
        d_types: list[data_types.DataType]|None=None, allow_data_type_fallback=False, include_data_loss=False
    ):
    output_directory = pathlib.Path(output_directory)
    if d_types is None:
        d_types = []

    # read input if needed
    if not isinstance(marker_intervals, pd.DataFrame):
        marker_intervals = pd.read_csv(marker_intervals, delimiter='\t', dtype={'marker_interval':int},index_col=['marker_interval','target'])
    if not all((c in marker_intervals or c in marker_intervals.index.names for c in ('marker_interval','target','start_timestamp','end_timestamp'))):
        raise ValueError('Provided marker intervals should contain the following columns: [marker_interval, target, start_timestamp, end_timestamp]')
    if not isinstance(gazes, dict):
        gazes = gaze_worldref.read_dict_from_file(gazes,validation_intervals)
    if not isinstance(poses, dict):
        poses = pose.read_dict_from_file(poses,validation_intervals)
    # prep targets
    targets_for_homography = {t_id: np.append(targets[t_id][0:2], distance_mm_for_homography) for t_id in targets}

    # for output
    out_df: pd.DataFrame = None

    # for each frame during analysis interval, determine offset
    # (angle) of gaze (each eye) to each of the targets
    for idx,iv in enumerate(validation_intervals):
        samples_per_frame = {k:v for (k,v) in gazes.items() if k>=iv[0] and k<=iv[1]}
        if not samples_per_frame:
            raise RuntimeError(f'There is no gaze data on the glassesValidator surface for validation interval (frames {idx[0]} to {idx[1]}), cannot proceed. This may be because there was no gaze during this interval or because the plane was not detected.')

        # check what data quality types we should output. Go with good defaults
        # first see what we have available
        d_have = data_types.get_available_data_types(samples_per_frame)
        # then determine, based on what user requests, what we will output
        d_types = data_types.select_data_types_to_use(d_types, d_have, allow_data_type_fallback)

        # prepare output data frame
        df_idx  = []
        idxer = pd.IndexSlice
        idxs = marker_intervals.loc[idx+1,:].index.to_frame().to_numpy()
        for dt in d_types:
            df_idx.append(np.vstack((np.full((idxs.shape[0],),idx+1,dtype='int'),idxs.shape[0]*[dt],idxs[:,0])).T)
        df_idx = pd.DataFrame(np.vstack(tuple(df_idx)),columns=[marker_intervals.index.names[0],'type',marker_intervals.index.names[1]])
        df  = pd.DataFrame(index=pd.MultiIndex.from_frame(df_idx.astype({marker_intervals.index.names[0]: 'int64', 'type': 'category', marker_intervals.index.names[1]: 'int64'})))

        # determine order in which targets were looked at
        for dt in d_types:
            df.loc[idxer[idx+1,dt,:],'order'] = np.argsort(marker_intervals.loc(axis=0)[idx+1,:]['start_timestamp'].to_numpy())+1
        if df['order'].dtype==np.dtype('float64'):  # ensure int if needed
            df['order'] = df['order'].astype(np.int64)

        # now compute
        for t in targets:
            if (idx+1,t) not in marker_intervals.index:
                continue
            frame_idxs  =           [k           for k,v in samples_per_frame.items() for _ in v]
            ts          = np.vstack([s.timestamp for  v in samples_per_frame.values() for s in v])
            st = marker_intervals.loc[(idx+1,t),'start_timestamp']
            et = marker_intervals.loc[(idx+1,t),  'end_timestamp']
            q_data = np.logical_and(ts>=st, ts<=et)

            offset = np.full((np.count_nonzero(q_data),len(d_types),2), np.nan)
            target_cam: dict[int,np.ndarray] = {}
            for idt,dt in enumerate(d_types):
                # get data
                fields = data_types.get_world_gaze_fields_for_data_type(dt)
                if fields is None:
                    continue
                if fields[0] is None:
                    ori     = np.zeros((ts.shape[0],3))
                else:
                    ori     = np.vstack([getattr(s,fields[0]) for v in samples_per_frame.values() for s in v])
                gazePlane   = np.vstack([getattr(s,fields[2]) for v in samples_per_frame.values() for s in v])
                if fields[1] is None:
                    if not dt==data_types.DataType.viewpos_vidpos_homography:
                        raise NotImplementedError(f'This field should be set, is a special case not implemented? Contact developer')
                    gaze    = np.hstack((gazePlane[:,0:2], np.full((gazePlane.shape[0],1),distance_mm_for_homography)))
                else:
                    gaze    = np.vstack([getattr(s,fields[1]) for v in samples_per_frame.values() for s in v])

                # compute
                out_idx = -1
                for i in range(len(ts)):
                    if not q_data[i]:
                        continue
                    out_idx += 1
                    frame_idx = frame_idxs[i]
                    if dt==data_types.DataType.viewpos_vidpos_homography:
                        # get vectors based on assumed viewing distance (from config), without using pose info
                        vGaze   = gaze[i,:]
                        vTarget = targets_for_homography[t]
                    else:
                        # use 3D vectors known given pose information
                        if frame_idx not in poses:
                            continue
                        if frame_idx not in target_cam:
                            target_cam[frame_idx] = poses[frame_idx].world_frame_to_cam(targets[t])

                        # get vectors from origin to target and to gaze point
                        vGaze   = gaze[i,:]            -ori[i,:]
                        vTarget = target_cam[frame_idx]-ori[i,:]

                    # get offset
                    ang2D               = transforms.angle_between(vTarget,vGaze)
                    # decompose in horizontal/vertical (in plane space)
                    onPlaneAngle        = math.atan2(gazePlane[i,1]-targets[t][1], gazePlane[i,0]-targets[t][0])
                    offset[out_idx,idt,:]= ang2D*np.array([math.cos(onPlaneAngle), math.sin(onPlaneAngle)])

            # special case for average of left and right eye
            if data_types.DataType.pose_left_right_avg in d_types:
                l_idx = d_types.index(data_types.DataType.pose_left_eye)
                r_idx = d_types.index(data_types.DataType.pose_right_eye)
                a_idx = d_types.index(data_types.DataType.pose_left_right_avg)
                offset[:,a_idx,:] = offset[:,[l_idx, r_idx],:].mean(axis=1)

            # compute data quality for this target
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # ignore warnings from np.nanmean, np.nanmedian and np.nanstd

                df.loc[idxer[idx+1,:,t],'acc_x'] = np.nanmedian(offset[:,:,0],axis=0)
                df.loc[idxer[idx+1,:,t],'acc_y'] = np.nanmedian(offset[:,:,1],axis=0)
                # NB: below is equivalent to computing offset of median gaze vector from target
                df.loc[idxer[idx+1,:,t],'acc'  ] = np.hypot(df.loc[idxer[idx+1,:,t],'acc_x'], df.loc[idxer[idx+1,:,t],'acc_y'])

                df.loc[idxer[idx+1,:,t],'rms_x'] = np.sqrt(np.nanmean(np.diff(offset[:,:,0],axis=0)**2))
                df.loc[idxer[idx+1,:,t],'rms_y'] = np.sqrt(np.nanmean(np.diff(offset[:,:,1],axis=0)**2))
                # NB: below is equivalent to np.sqrt(np.nanmean(xdiff+ydiff))
                df.loc[idxer[idx+1,:,t],'rms'  ] = np.hypot(df.loc[idxer[idx+1,:,t],'rms_x'], df.loc[idxer[idx+1,:,t],'rms_y'])

                df.loc[idxer[idx+1,:,t],'std_x'] = np.nanstd(offset[:,:,0],ddof=0,axis=0)
                df.loc[idxer[idx+1,:,t],'std_y'] = np.nanstd(offset[:,:,1],ddof=0,axis=0)
                df.loc[idxer[idx+1,:,t],'std'  ] = np.hypot(df.loc[idxer[idx+1,:,t],'std_x'], df.loc[idxer[idx+1,:,t],'std_y'])

                if include_data_loss:
                    df.loc[idxer[idx+1,:,t],'data_loss'] = np.sum(np.isnan(offset[:,:,0]),axis=0)/offset.shape[0]

        if out_df is None:
            out_df = df
        else:
            out_df = pd.concat((out_df,df))

    out_df.to_csv(output_directory / filename, mode='w', header=True, sep='\t', na_rep='nan', float_format="%.6f")