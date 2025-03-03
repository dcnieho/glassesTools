import math
import numpy as np
import pandas as pd
import typing
import pathlib

import I2MC
import matplotlib.pyplot as plt

from . import gaze_worldref, naming

# run I2MC on data projected to a plane
def from_plane_gaze(
        gazes: str|pathlib.Path|dict[int, list[gaze_worldref.Gaze]],
        classification_intervals: list[list[int]],
        output_directory: str|pathlib.Path,
        I2MC_settings_override: dict[str,typing.Any]=None,
        filename_stem: str = naming.fixation_classification_prefix,
        do_plot = True,
        plot_limits: list[list[float]] = None
    ):
    output_directory = pathlib.Path(output_directory)

    # read input if needed
    if not isinstance(gazes,dict):
        gazes = gaze_worldref.read_dict_from_file(gazes)

    # set I2MC options
    opt = {'xres': None, 'yres': None}  # dummy values for required options
    opt['missingx']         = math.nan
    opt['missingy']         = math.nan
    opt['maxdisp']          = 50        # mm
    opt['windowtimeInterp'] = .25       # s
    opt['maxMergeDist']     = 20        # mm
    opt['maxMergeTime']     = 81        # ms
    opt['minFixDur']        = 50        # ms

    # decide what sampling frequency to tell I2MC about. It doesn't work with varying sampling frequency, nor
    # any random sampling frequency. For our purposes, getting it right is not important (internally I2MC only
    # uses sampling frequency for converting some of the time units to samples, other things are taken directly
    # from the time signal). So, we have working I2MC settings for a few sampling frequencies, and just choose
    # the nearest based on empirically determined sampling frequency.
    ts          = np.array([s.timestamp for v in gazes.values() for s in v])
    recFreq     = np.round(np.mean(1000./np.diff(ts)))    # Hz
    knownFreqs  = [30., 50., 60., 90., 120., 200.]
    opt['freq'] = knownFreqs[np.abs(knownFreqs - recFreq).argmin()]
    if opt['freq']==200.:
        pass    # defaults are good
    elif opt['freq']==120.:
        opt['downsamples']      = [2, 3, 5]
        opt['chebyOrder']       = 7
    elif opt['freq'] in [50., 60.]:
        opt['downsamples']      = [2, 5]
        opt['downsampFilter']   = False
    else:
        # 90 Hz, 30 Hz
        opt['downsamples']      = [2, 3]
        opt['downsampFilter']   = False

    # apply setting overrides from caller, if any
    if I2MC_settings_override:
        for k in I2MC_settings_override:
            if I2MC_settings_override[k] is not None:
                opt[k] = I2MC_settings_override[k]

    # collect data
    qHasLeft        = np.any(np.logical_not(np.isnan([s.gazePosPlane2DLeft               for v in gazes.values() for s in v])))
    qHasRight       = np.any(np.logical_not(np.isnan([s.gazePosPlane2DRight              for v in gazes.values() for s in v])))
    qHasRay         = np.any(np.logical_not(np.isnan([s.gazePosPlane2D_vidPos_ray        for v in gazes.values() for s in v])))
    qHasHomography  = np.any(np.logical_not(np.isnan([s.gazePosPlane2D_vidPos_homography for v in gazes.values() for s in v])))
    for idx,iv in enumerate(classification_intervals):
        gazesToClassify = {k:v for (k,v) in gazes.items() if k>=iv[0] and (iv[1]==-1 or k<=iv[1])}
        # Doing detection on the world data if available is good, but we should plot using the ray (if
        # available) or homography data, as that corresponds to the gaze visualization provided in the
        # software, and for some recordings/devices the world-based coordinates can be far off.
        if qHasRay:
            ray_x  = np.array([s.gazePosPlane2D_vidPos_ray[0] for v in gazesToClassify.values() for s in v])
            ray_y  = np.array([s.gazePosPlane2D_vidPos_ray[1] for v in gazesToClassify.values() for s in v])
        elif qHasHomography:
            homography_x  = np.array([s.gazePosPlane2D_vidPos_homography[0] for v in gazesToClassify.values() for s in v])
            homography_y  = np.array([s.gazePosPlane2D_vidPos_homography[1] for v in gazesToClassify.values() for s in v])

        data = {}
        data['time'] = np.array([s.timestamp for v in gazesToClassify.values() for s in v])
        qNeedRecalcFix = False
        if qHasLeft and qHasRight:
            # prefer using separate left and right eye signals, if available. Better I2MC robustness
            data['L_X']  = np.array([s.gazePosPlane2DLeft[0]  for v in gazesToClassify.values() for s in v])
            data['L_Y']  = np.array([s.gazePosPlane2DLeft[1]  for v in gazesToClassify.values() for s in v])
            data['R_X']  = np.array([s.gazePosPlane2DRight[0] for v in gazesToClassify.values() for s in v])
            data['R_Y']  = np.array([s.gazePosPlane2DRight[1] for v in gazesToClassify.values() for s in v])
            qNeedRecalcFix = True
        elif qHasRay:
            data['average_X']  = ray_x
            data['average_Y']  = ray_y
        elif qHasHomography:
            data['average_X']  = homography_x
            data['average_Y']  = homography_y
        else:
            raise RuntimeError('No data available to process')

        # run event classification to find fixations
        fixations, data_I2MC, par_I2MC = I2MC.I2MC(data, opt, False)

        # replace gaze data used for classification with gaze position on scene video
        # see note above for why
        if qNeedRecalcFix:
            data_I2MC = data_I2MC.drop(columns=['L_X','L_Y','R_X','R_Y'],errors='ignore')
            data_I2MC['average_X'] = ray_x if qHasRay else homography_x
            data_I2MC['average_Y'] = ray_y if qHasRay else homography_y
            # recalculate fixation positions based on gaze position on video data
            fixations = I2MC.get_fixations(data_I2MC['finalweights'].array, data_I2MC['time'].array, data_I2MC['average_X'], data_I2MC['average_Y'], data_I2MC['average_missing'], par_I2MC)

        # store to file
        fix_df = pd.DataFrame(fixations)
        fix_df.to_csv(output_directory / f'{filename_stem}_interval_{idx+1:02d}.tsv',
                      mode='w', na_rep='nan', sep='\t', index=False, float_format='%.3f')

        # make timeseries plot of gaze data with fixations
        if do_plot:
            f = I2MC.plot.data_and_fixations(data_I2MC, fixations, fix_as_line=True, unit='mm', res=plot_limits)
            plt.gca().invert_yaxis()
            f.savefig(str(output_directory / f'{filename_stem}_interval_{idx+1:02d}.png'))
            plt.close(f)