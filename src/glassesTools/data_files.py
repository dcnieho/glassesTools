import numpy as np
import pandas as pd
import polars as pl
import pathlib
import importlib
from typing import Any, Optional
from collections import defaultdict


def getColumnLabels(lbl,N=3):
    if N<=3:
        return [lbl+'_%s' % (chr(c)) for c in range(ord('x'), ord('x')+N)]
    elif N==9:
        return [lbl+'[%d,%d]' % (r,c) for r in range(3) for c in range(3)]
    else:
        raise ValueError(f'N input should be <=3 or 9, was {N}')

def noneIfAnyNan(vals):
    if not np.any(np.isnan(vals)):
        return vals
    else:
        return None

def allNanIfNone(vals, numel):
    if vals is None:
        return np.full((numel,), np.nan)
    else:
        return vals


# read coordinate files (e.g. marker files, which have the colums ID, x, y, rotation_angle)
def _read_coord_file_impl(file):
    return pd.read_csv(str(file), index_col='ID')

def read_coord_file(file, package_to_read_from=None):
    # if directory is not provided, try to read the file from the package resources instead
    if package_to_read_from:
        with importlib.resources.path(package_to_read_from, file) as p:
            return _read_coord_file_impl(p)
    else:
        if file.is_file():
            return _read_coord_file_impl(file)
        else:
            return None


def uncompress_columns(cols_compressed: dict[str, int]):
    return [getColumnLabels(c,N) if (N:=cols_compressed[c])>1 else [c] for c in cols_compressed]

def _get_col_name_with_suffix(base:str, suf:str):
    if not suf:
        return base
    return base + '_' + suf

def read_file(fileName               : str|pathlib.Path,
              object                 : Any,
              drop_if_all_nan        : bool,
              none_if_any_nan        : bool,
              as_list_dict           : bool,
              make_ori_ts_fridx      : bool,
              episodes               : Optional[list[list[int]]] = None,
              ts_fridx_field_suffixes: list[str]                 = None,
              subset_var                                         = 'frame_idx'):

    # interrogate destination object
    cols_compressed: dict[str, int] = object._columns_compressed
    dtypes         : dict[str, Any] = object._non_float

    # read file and select, if wanted
    df          = pd.read_csv(fileName, delimiter='\t', index_col=False, dtype=defaultdict(lambda: float, **dtypes))
    if episodes:
        sel = (df[subset_var] >= episodes[0][0]) & (df[subset_var] <= episodes[0][1])
        for e in episodes[1:]:
            sel |= (df[subset_var] >= e[0]) & (df[subset_var] <= e[1])
        df = df[sel]

    # figure out what the data columns are
    cols_uncompressed = uncompress_columns(cols_compressed)

    # drop rows where are all data columns are nan
    if drop_if_all_nan:
        df = df.dropna(how='all',subset=[c for cs in cols_uncompressed if len(cs)>1 for c in cs])

    # group columns into numpy arrays, optionally insert None if missing
    for c,ac in zip(cols_compressed,cols_uncompressed):
        if len(ac)==1:
            continue    # nothing to do, would just copy column to itself
        elif ac:
            if not any([a in df.columns for a in ac]):
                continue
            if none_if_any_nan:
                df[c] = [noneIfAnyNan(x) for x in df[ac].values]  # make list of numpy arrays, or None if there are any NaNs in the array
            else:
                df[c] = [             x  for x in df[ac].values]  # make list of numpy arrays
        else:
            df[c] = None

    # keep only the columns we want (this also puts them in the right order even if that doesn't matter since we use kwargs to construct objects)
    df = df[[c for c in cols_compressed.keys() if c in df.columns]]

    # if we have multiple timestamps of frame_idxs, make sure we keep a copy of the original one
    if make_ori_ts_fridx:
        # make copies of original
        df['frame_idx_ori'] = df['frame_idx']
        if 'timestamp' in df.columns:
            df['timestamp_ori'] = df['timestamp']
    # now put requested into normal timestamp column, if wanted
    if make_ori_ts_fridx and ts_fridx_field_suffixes:
        copied = False
        for suf in ts_fridx_field_suffixes: # these are in order of preference
            field = _get_col_name_with_suffix('frame_idx',suf)
            if field not in df.columns:
                continue
            df['frame_idx'] = df[field]
            if 'timestamp' in df.columns:
                df['timestamp'] = df[_get_col_name_with_suffix('timestamp',suf)]
            copied = True
            break
        assert copied, "None of the specified suffixes were found, can't continue"

    if as_list_dict:
        obj_list = [object(**kwargs) for kwargs in df.to_dict(orient='records')]

        # organize into dict by frame index
        objs = {}
        for k,v in zip(df[subset_var],obj_list):
            objs.setdefault(k, []).append(v)
    else:
        objs = {idx:object(**kwargs) for idx,kwargs in zip(df[subset_var].values,df.to_dict(orient='records'))}
    return objs, df[subset_var].max()

def write_array_to_file(objects             : list[Any] | dict[int,list[Any]],
                        fileName            : str|pathlib.Path,
                        cols_compressed     : dict[str, int],
                        drop_all_nan_cols   : list[str]         = None,
                        skip_all_nan        : bool              = False):
    if not objects:
        return

    if drop_all_nan_cols is None:
        drop_all_nan_cols = []

    if isinstance(objects, dict):
        # flatten
        objects = [o for olist in objects.values() for o in olist]

    records = [{k:getattr(p,k) for k in vars(p) if not k.startswith('_')} for p in objects]
    df = pd.DataFrame.from_records(records)

    # unpack array columns
    cols_uncompressed = uncompress_columns(cols_compressed)
    for c,ac in zip(cols_compressed,cols_uncompressed):
        if len(ac)>1:
            df[ac] = np.vstack([allNanIfNone(v,len(ac)).flatten() for v in df[c].values])

    # if wanted, drop specific columns for which all rows are nan
    if drop_all_nan_cols:
        df = df.drop([c for c in drop_all_nan_cols if c in df and df[c].isnull().all()], axis='columns')

    # if we have filled _ori timestamp and frame_idx columns, copy them back into plain timestamp
    # and frame_idx columns. NB: the _ori columns will be removed because they're not listed in
    # the possible file columns
    if 'timestamp_ori' in df.columns and not df['timestamp_ori'].isnull().all():
        df['timestamp'] = df['timestamp_ori']
    if 'frame_idx_ori' in df.columns and not df['frame_idx_ori'].isnull().all():
        df['frame_idx'] = df['frame_idx_ori']

    # keep only columns to be written out and order them correctly
    df = df[[c for cs in cols_uncompressed for c in cs if c in df.columns]]

    # drop rows where are all data columns are nan
    if skip_all_nan:
        df = df.dropna(how='all', subset=[c for cs in cols_uncompressed if len(cs)>1 for c in cs])

    # convert to polars as that library saves to file waaay faster
    df = pl.from_pandas(df)
    df.write_csv(fileName, separator='\t', null_value='nan', float_precision=8)
