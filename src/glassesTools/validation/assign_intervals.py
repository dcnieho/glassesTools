import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

from .. import gaze_worldref, marker, naming

# assign intervals to targets based on distance
def distance(
        targets: dict[int, np.ndarray],
        fixations: str|pathlib.Path|pd.DataFrame,
        do_global_shift = True,
        max_dist_fac = .5
    ) -> tuple[pd.DataFrame, pd.DataFrame|None]:

    # read input if needed
    if not isinstance(fixations,pd.DataFrame):
        fixations = pd.read_csv(fixations,sep='\t',index_col=False)

    # for each target, find closest fixation
    minDur      = 100       # ms
    used        = np.zeros((fixations['start'].size),dtype='bool')
    selected    = np.empty((len(targets),),dtype='int')
    selected[:] = -999

    t_x = np.array([targets[t][0] for t in targets])
    t_y = np.array([targets[t][1] for t in targets])
    off_f_x = off_f_y = off_t_x = off_t_y = 0.
    if do_global_shift:
        # first, center the problem. That means determine and remove any overall shift from the
        # data and the targets, to improve robustness of assigning fixations points to targets.
        # Else, if all data is e.g. shifted up by more than half the distance between
        # validation targets, target assignment would fail
        off_f_x = fixations['xpos'].median()
        off_f_y = fixations['ypos'].median()
        off_t_x = t_x.mean()
        off_t_y = t_y.mean()

    # we furthermore do not assign a fixation to a target if the closest fixation is more than
    # max_dist_fac the intertarget distance away
    # determine intertarget distance, if possible
    dist_lim = np.inf
    if len(t_x)>1:
        # arbitrarily take first target and find closest target to it
        dist = np.hypot(t_x[0]-t_x[1:], t_y[0]-t_y[1:])
        min_dist = dist.min()
        if min_dist > 0:
            dist_lim = min_dist*max_dist_fac

    for i,t in zip(range(len(targets)),targets):
        if np.all(used):
            # all fixations used up, can't assign anything to remaining targets
            continue
        # select fixation
        dist        = np.hypot(fixations['xpos']-off_f_x-(targets[t][0]-off_t_x), fixations['ypos']-off_f_y-(targets[t][1]-off_t_y))
        dist[used]  = np.inf                    # make sure fixations already bound to a target are not used again
        dist[fixations['dur']<minDur] = np.inf  # make sure that fixations that are too short are not selected
        iFix        = np.argmin(dist)
        if dist[iFix]<=dist_lim:
            selected[i] = iFix
            used[iFix]  = True

    # prep return values
    selected_intervals = pd.DataFrame(columns=['xpos','ypos','startT','endT'])  # sets which columns to copy
    selected_intervals.index.name = 'target'
    for i,t in zip(range(len(targets)),targets):
        if selected[i]==-999:
            continue
        selected_intervals.loc[t] = fixations.iloc[selected[i]]
    other_intervals = fixations.loc[np.logical_not(used),['xpos','ypos','startT']]
    return selected_intervals, None if other_intervals.empty else other_intervals

def dynamic_markers(
        marker_observations_per_target: dict[int, pd.DataFrame],    # {target: dataframe}
        markers_per_target: dict[int,list[marker.MarkerID]],        # {target: [marker,...]}, one or multiple markers per target
        timestamps_file: str|pathlib.Path,
        episode: list[int],
        skip_first_duration: float,
        max_gap_duration: int,
        min_duration: int
    ) -> tuple[pd.DataFrame, None]:
    # also need frame timestamps because the intervals returned by this function should be expressed in recording time, not as frame indices.
    timestamps  = pd.read_csv(timestamps_file, delimiter='\t', index_col='frame_idx')
    ts_col      = 'timestamp_stretched' if 'timestamp_stretched' in timestamps else 'timestamp'

    # make local copy of marker_observations, containing only the current episode
    marker_observations_per_target = {t:mo.loc[episode[0]:episode[1],:] for t,mo in marker_observations_per_target.items()}
    # check we have data for at least one of the markers for a given target
    for t in marker_observations_per_target:
        if marker_observations_per_target[t].empty:
            missing_str  = '\n- '.join([marker.marker_ID_to_str(m) for m in markers_per_target[t]])
            raise RuntimeError(f'None of the markers for target {t} were observed during the episode from frame {episode[0]} to frame {episode[1]}:\n- {missing_str}')

    # marker presence signal only contains marker detections (True). We need to fill the gaps in between detections with False (not detected) so we have a continuous signal without gaps
    marker_observations_per_target = {t: marker.expand_detection(marker_observations_per_target[t], fill_value=False) for t in marker_observations_per_target}

    # for each target, see when it is presented using the marker presence signal
    selected_intervals = pd.DataFrame(columns=['startT','endT'])
    selected_intervals.index.name = 'target'
    for t in marker_observations_per_target:
        start, end = marker.get_appearance_starts_ends(marker_observations_per_target[t], max_gap_duration, min_duration)
        if start.size==0:
            continue
        # in case there are multiple (e.g. spotty detection), choose longest
        durs = np.array(end)-np.array(start)+1
        maxi = np.argmax(durs)
        ts = timestamps.loc[[start[maxi], end[maxi]],ts_col].to_numpy()
        ts[0] += skip_first_duration
        if ts[0]>=ts[1]:
            continue
        selected_intervals.loc[t] = ts
    return selected_intervals, None

def plot(
        selected_intervals: pd.DataFrame,
        other_intervals: pd.DataFrame|None,
        targets: dict[int, np.ndarray],
        gazes: str|pathlib.Path|dict[int, list[gaze_worldref.Gaze]],
        episode: list[int],
        output_directory: str|pathlib.Path,
        filename_stem: str = naming.fixation_assignment_prefix,
        iteration = 0,
        background_image: tuple[np.ndarray, list[float]] = None,    # (image, extent in mm [l r t b])
        plot_limits: list[list[float]] = None
    ):
    output_directory = pathlib.Path(output_directory)
    # if we do not have x and y positions for the gaze intervals, make them
    if 'xpos' not in selected_intervals.columns or (other_intervals is not None and 'xpos' not in other_intervals.columns):
        # read input if needed
        if not isinstance(gazes,dict):
            gazes = gaze_worldref.read_dict_from_file(gazes)
        if 'xpos' not in selected_intervals.columns:
            samples_per_frame = {k:v for (k,v) in gazes.items() if k>=episode[0] and k<=episode[1]}
            has_ray = np.any(np.logical_not(np.isnan([s.gazePosPlane2D_vidPos_ray for v in gazes.values() for s in v])))
            field = 'gazePosPlane2D_vidPos_ray' if has_ray else 'gazePosPlane2D_vidPos_homography'
            for t in selected_intervals.index:
                st,et = selected_intervals.loc[t,['startT','endT']].to_numpy()
                data = [getattr(s,field) for v in samples_per_frame.values() for s in v if s.timestamp>=st and s.timestamp<=et]
                if data:
                    gaze = np.vstack(data)
                    selected_intervals.loc[t,['xpos','ypos']] = np.nanmedian(gaze,axis=0)
        if other_intervals is not None and 'xpos' not in other_intervals.columns:
            for t in other_intervals.index:
                st,et = other_intervals.loc[t,['startT','endT']].to_numpy()
                data = [getattr(s,field) for v in samples_per_frame.values() for s in v if s.timestamp>=st and s.timestamp<=et]
                if data:
                    gaze = np.vstack(data)
                    other_intervals.loc[t,['xpos','ypos']] = np.nanmedian(gaze,axis=0)
    # make plot of data overlaid on poster, and show for each target which interval was selected
    # prep data
    if other_intervals is not None:
        all_intervals = pd.concat((selected_intervals.set_index('startT'), other_intervals.set_index('startT')), join='inner').sort_index()
    else:
        all_intervals = selected_intervals.set_index('startT').sort_index()

    f       = plt.figure(dpi=300)
    imgplot = plt.imshow(background_image[0],extent=background_image[1],alpha=.5)
    # draw all intervals
    plt.plot(all_intervals['xpos'],all_intervals['ypos'],'b-')
    plt.plot(all_intervals['xpos'],all_intervals['ypos'],'go')
    # draw target matching
    for t, row in selected_intervals.iterrows():
        plt.plot([row['xpos'], targets[t][0]], [row['ypos'], targets[t][1]],'r-')

    # cosmetics
    plt.xlabel('mm')
    plt.ylabel('mm')
    if plot_limits is not None:
        plt.xlim(plot_limits[0])
        plt.ylim(plot_limits[1])
    plt.gca().invert_yaxis()

    f.savefig(output_directory / f'{filename_stem}_interval_{iteration+1:02d}.png')
    plt.close(f)

def to_tsv(
        selected_intervals: pd.DataFrame,
        output_directory: str|pathlib.Path,
        filename_stem: str = naming.fixation_assignment_prefix,
        iteration = 0,
    ):
    output_directory = pathlib.Path(output_directory)
    # store selected intervals
    selected_intervals = selected_intervals \
                            .drop(columns=[c for c in ('xpos','ypos') if c in selected_intervals.columns]) \
                            .rename(columns={'startT': 'start_timestamp', 'endT': 'end_timestamp'})
    selected_intervals.insert(0, 'marker_interval', iteration+1)

    selected_intervals.to_csv(
        output_directory / f'{filename_stem}.tsv',
        mode='w' if iteration==0 else 'a',
        header=iteration==0,
        sep='\t', na_rep='nan', float_format="%.3f")