import numpy as np
import pandas as pd
import itertools
import pathlib
from typing import Optional, Any
from collections import defaultdict


def getXYZLabels(stringList,N=3):
    if type(stringList) is not list:
        stringList = [stringList]
    return list(itertools.chain(*[[s+'_%s' % (chr(c)) for c in range(ord('x'), ord('x')+N)] for s in stringList]))

def noneIfAnyNan(vals):
    if not np.any(np.isnan(vals)):
        return vals
    else:
        return None

def allNanIfNone(vals,numel):
    if vals is None:
        return np.array([np.nan for _ in range(numel)])
    else:
        return vals


def readfile(fileName: str|pathlib.Path,
             array_cols: list[str],
             col_order : list[str],
             none_if_any_nan: bool,
             object: Any,

             start:Optional[int]=None,
             end:Optional[int]=None,
             subset_var='frame_idx'):

    df          = pd.read_csv(str(fileName), delimiter='\t', index_col=False, dtype=defaultdict(lambda: float, frame_idx=int, poseOk=bool, poseNMarker=int, homographyNMarker=int))
    if start is not None and end is not None:
        df = df[(df[subset_var] >= start) & (df[subset_var] <= end)]

    # figure out what the data columns are
    array_cols = ('poseRvec','poseTvec','homography[')
    all_cols   = tuple([c for c in df.columns if col in c] for col in array_cols)

    # drop rows that are all data columns are nan
    df = df.dropna(how='all',subset=[c for cs in all_cols for c in cs])

    # group columns into numpy arrays, insert None if missing
    for c,ac in zip(array_cols,all_cols):
        if ac:
            if none_if_any_nan:
                df[c] = [noneIfAnyNan(x) for x in df[ac].values]  # make list of numpy arrays, or None if there are any NaNs in the array
            else:
                df[c] = [             x  for x in df[ac].values]  # make list of numpy arrays
        else:
            df[c] = None

    # clean up so we can assign into gaze objects directly
    lookup = {'frame_idx'           :'frameIdx',
                'poseOk'              :'poseOk',
                'poseNMarker'         :'nMarkers',
                'poseRvec'            :'rVec',
                'poseTvec'            :'tVec',
                'homographyNMarker'   :'nMarkersH',
                'homography['         :'hMat'}
    # keep only the columns we want and ensure that they are in the right order (doesn't matter since we use kwargs, still doesn't hurt either)
    df = df[lookup.keys()]
    # ensure they have names matching the input parameters of the Pose constructor
    df = df.rename(columns=lookup)

    poses = {idx:Pose(**kwargs) for idx,kwargs in zip(df['frameIdx'].values,df.to_dict(orient='records'))}
    return poses