import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

import I2MC

from .. import naming

# assign fixations to targets based on distance
def distance(
        targets: dict[int, np.ndarray],
        fixations: str|pathlib.Path|pd.DataFrame,
        output_directory: str|pathlib.Path,
        do_global_shift = True, max_dist_fac = .5,
        filename_stem: str = naming.fixation_assignment_prefix,
        iteration = 0,
        background_image: tuple[np.ndarray, list[float]] = None,    # (image, extent in mm [l r t b])
        plot_limits: list[list[float]] = None
    ):
    output_directory = pathlib.Path(output_directory)

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

    # make plot of data overlaid on poster, and show for each target which fixation
    # was selected
    if background_image is not None:
        f       = plt.figure(dpi=300)
        imgplot = plt.imshow(background_image[0],extent=background_image[1],alpha=.5)
        plt.plot(fixations['xpos'],fixations['ypos'],'b-')
        plt.plot(fixations['xpos'],fixations['ypos'],'go')
        if plot_limits is not None:
            plt.xlim(plot_limits[0])
            plt.ylim(plot_limits[1])
        plt.gca().invert_yaxis()
        for i,t in zip(range(len(targets)),targets):
            if selected[i]==-999:
                continue
            plt.plot([fixations['xpos'][selected[i]], targets[t][0]], [fixations['ypos'][selected[i]], targets[t][1]],'r-')

        plt.xlabel('mm')
        plt.ylabel('mm')

        f.savefig(output_directory / f'{filename_stem}_interval_{iteration+1:02d}.png')
        plt.close(f)

    # store selected intervals
    df = pd.DataFrame()
    df.index.name = 'target'
    for i,t in zip(range(len(targets)),targets):
        if selected[i]==-999:
            continue
        df.loc[t,'marker_interval'] = iteration+1
        df.loc[t,'start_timestamp'] = fixations['startT'][selected[i]]
        df.loc[t,  'end_timestamp'] = fixations[  'endT'][selected[i]]

    df.to_csv(output_directory / f'{filename_stem}.tsv',
              mode='w' if iteration==0 else 'a',
              header=iteration==0,
              sep='\t', na_rep='nan', float_format="%.3f")