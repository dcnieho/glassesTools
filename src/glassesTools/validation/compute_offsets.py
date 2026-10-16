import pathlib
import numpy as np
import pandas as pd
import warnings

from ETDQualitizer import _frechet_median_on_sphere, data_loss_from_invalid, rms_s2s, std

from .. import data_types, gaze_worldref, naming, pose


def compute(
        gazes: str|pathlib.Path|dict[int, list[gaze_worldref.Gaze]],
        poses: str|pathlib.Path|dict[int, pose.Pose]|dict[int, list[pose.Pose]],
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
        gazes = gaze_worldref.read_dict_from_file(gazes, validation_intervals, ts_column_suffixes=['VOR',''])
    if not isinstance(poses, dict):
        poses = pose.read_list_dict_from_file(poses,validation_intervals, ts_column_suffixes=['VOR',''])
    # prep targets
    targets_for_homography = {t_id: np.append(targets[t_id][0:2], distance_mm_for_homography) for t_id in targets}

    # for output
    out_df: pd.DataFrame = None

    # for each frame during analysis interval, determine offset
    # (angle) of gaze (each eye) to each of the targets
    for idx,iv in enumerate(validation_intervals):
        samples_per_frame = {k:v for k,v in gazes.items() if iv[0] <= k <= iv[1]}
        if not samples_per_frame:
            raise RuntimeError(f'There is no gaze data on the glassesValidator surface for validation interval (frames {iv[0]} to {iv[1]}), cannot proceed. This may be because there was no gaze during this interval or because the plane was not detected.')

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
            ts          = np.asarray([s.timestamp for v in samples_per_frame.values() for s in v], dtype=float)
            st = marker_intervals.loc[(idx+1,t),'start_timestamp']
            et = marker_intervals.loc[(idx+1,t),  'end_timestamp']
            q_data = np.logical_and(ts>=st, ts<=et)

            _, _, relative_vectors = data_types.calculate_target_relative_gaze_vectors(samples_per_frame, poses, targets, d_types, targets_for_homography, distance_mm_for_homography)

            # compute data quality for this target
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # ignore warnings from np.nanmean, np.nanmedian and np.nanstd

                for dt in d_types:
                    vectors = relative_vectors[t][dt][q_data]
                    azi, ele = data_types.target_relative_vectors_to_fick(vectors)
                    median_vector = _frechet_median_on_sphere(vectors)
                    median_azi, median_ele = data_types.target_relative_vectors_to_fick(median_vector)
                    acc = np.degrees(np.arctan2(np.hypot(median_vector[0], median_vector[1]), median_vector[2]))
                    rms, rms_x, rms_y = rms_s2s(azi, ele)
                    std_total, std_x, std_y = std(azi, ele)
                    row = idxer[idx+1,dt,t]
                    df.loc[row,'acc_x'] = median_azi
                    df.loc[row,'acc_y'] = median_ele
                    df.loc[row,'acc'] = acc
                    df.loc[row,'rms_x'] = rms_x
                    df.loc[row,'rms_y'] = rms_y
                    df.loc[row,'rms'] = rms
                    df.loc[row,'std_x'] = std_x
                    df.loc[row,'std_y'] = std_y
                    df.loc[row,'std'] = std_total

                    if include_data_loss:
                        df.loc[row,'data_loss'] = data_loss_from_invalid(azi, ele) if len(vectors) else np.nan

        if out_df is None:
            out_df = df
        else:
            out_df = pd.concat((out_df,df))

    out_df.to_csv(output_directory / filename, mode='w', header=True, sep='\t', na_rep='nan', float_format="%.6f")
