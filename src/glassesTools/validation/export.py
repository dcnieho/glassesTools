import csv
import pathlib
import pandas as pd
import typing

from .. import data_types

def _read_data_quality_file(file: pathlib.Path) -> pd.DataFrame:
    skiprows = []
    with file.open(encoding='utf-8', newline='') as input_file:
        reader = csv.reader(input_file, delimiter='\t')
        header = next(reader, [])
        first_data_row = next(reader, None)
    if first_data_row is not None and 'marker_interval' in header:
        try:
            float(first_data_row[header.index('marker_interval')])
        except (IndexError, ValueError):
            skiprows = [1]
    return pd.read_csv(file, delimiter='\t', skiprows=skiprows)

def collect_data_quality(rec_dirs: typing.Sequence[str | pathlib.Path], file_name: str|dict[str,str]='dataQuality.tsv', col_for_parent=None):
    # 1. collect all data quality metrics from the provided directories
    rec_files: list = []
    idx_vals = ['recording']
    if isinstance(file_name,dict):
        for f in file_name:
            for d in rec_dirs:
                f_path = pathlib.Path(d)/file_name[f]
                if not f_path.is_file():
                    continue
                kwargs = {'recording': f_path.parent.name, 'plane': f}
                if col_for_parent:
                    kwargs[col_for_parent] = f_path.parent.parent.name
                rec_files.append((f_path,kwargs))
        idx_vals.append('plane')
        if col_for_parent:
            idx_vals.insert(0,col_for_parent)
    else:
        rec_files = [(pathlib.Path(rec)/file_name,{'recording': rec.name}) for rec in rec_dirs]
        rec_files = [f for f in rec_files if f[0].is_file()]
    if not rec_files:
        return None, None, None
    df = pd.concat((_read_data_quality_file(rec[0]).assign(**rec[1]) for rec in rec_files), ignore_index=True)
    if df.empty:
        return None, None, None
    # set indices
    df = df.set_index(idx_vals+['marker_interval','type','target'])
    # change type index into enum
    typeIdx = df.index.names.index('type')
    df.index = df.index.set_levels(pd.CategoricalIndex([getattr(data_types.DataType,x) for x in df.index.levels[typeIdx]]),level='type')

    # see what we have
    dq_types = sorted(list(df.index.levels[typeIdx]), key=lambda dq: dq.value)
    targets  = list(df.index.levels[df.index.names.index('target')])

    # good default selection of dq type to export
    if data_types.DataType.pose_vidpos_ray in dq_types:
        default_dq_type = data_types.DataType.pose_vidpos_ray
    elif data_types.DataType.pose_vidpos_homography in dq_types:
        default_dq_type = data_types.DataType.pose_vidpos_homography
    else:
        # ultimate fallback, just set first available as the one to export
        default_dq_type = dq_types[0]

    return df, default_dq_type, targets

def summarize_and_store_data_quality(df: pd.DataFrame, output_file_or_dir: str | pathlib.Path, dq_types: list[data_types.DataType], targets: list[int], average_over_targets = False, include_data_loss = False, include_unit_header_row = True):
    dq_types_have = sorted(list(df.index.levels[df.index.names.index('type')]), key=lambda dq: dq.value)
    targets_have  = list(df.index.levels[df.index.names.index('target')])

    # remove unwanted types of data quality
    dq_types_sel = [dq in dq_types for dq in dq_types_have]
    if not all(dq_types_sel):
        df = df.drop(index=[dq for i,dq in enumerate(dq_types_have) if not dq_types_sel[i]], level='type')
    # remove unwanted targets
    targets_sel = [t in targets for t in targets_have]
    if not all(targets_sel):
        df = df.drop(index=[t for i,t in enumerate(targets_have) if not targets_sel[i]], level='target')
    # remove optional data-quality measures if not wanted
    if not include_data_loss and 'data_loss' in df.columns:
        df = df.drop(columns='data_loss')
    # average data if wanted
    if average_over_targets:
        gb = df.drop(columns='order').groupby([n for n in df.index.names if n!='target'],observed=True)
        count = gb.count()
        df = gb.mean()
        # add number of targets count (there may be some missing data)
        df.insert(0,'num_targets',count['acc'])

    # store
    output_file_or_dir = pathlib.Path(output_file_or_dir)
    if output_file_or_dir.is_dir():
        output_file_or_dir = output_file_or_dir / 'dataQuality.tsv'
    if include_unit_header_row:
        df = df.reset_index()
        column_info = {
            'session': 'session name',
            'recording': 'recording name',
            'plane': 'validation plane name',
            'marker_interval': 'validation interval number',
            'type': 'gaze data type',
            'target': 'validation target ID',
            'num_targets': 'number of included targets',
            'order': 'target presentation order within interval',
            'plane_target_direction_x': 'target azimuth (Fick) relative to validation plane center target (deg)',
            'plane_target_direction_y': 'target elevation (Fick) relative to validation plane center target (deg)',
            'acc_x': 'accuracy azimuth (Fick) component (deg)',
            'acc_y': 'accuracy elevation (Fick) component (deg)',
            'acc': 'accuracy magnitude (deg)',
            'rms_x': 'RMS precision, azimuth (Fick) (deg)',
            'rms_y': 'RMS precision, elevation (Fick) (deg)',
            'rms': 'RMS precision (deg)',
            'std_x': 'standard deviation, azimuth (Fick) (deg)',
            'std_y': 'standard deviation, elevation (Fick) (deg)',
            'std': 'standard deviation (deg)',
            'plane_distance': 'camera-to-validation-plane distance (mm)',
            'video_target_direction_x': 'target azimuth (Fick) relative to scene camera center (deg)',
            'video_target_direction_y': 'target elevation (Fick) relative to scene camera center (deg)',
            'data_loss': 'data loss (%)',
        }
        with output_file_or_dir.open('w', encoding='utf-8', newline='') as output:
            writer = csv.writer(output, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(df.columns)
            writer.writerow([column_info.get(column, '') for column in df.columns])
            df.to_csv(output, header=False, index=False, sep='\t', na_rep='nan', float_format="%.6f")
    else:
        df.to_csv(output_file_or_dir, mode='w', header=True, sep='\t', na_rep='nan', float_format="%.6f")

def export_data_quality(rec_dirs: list[str | pathlib.Path], output_file_or_dir: str | pathlib.Path, dq_types: list[data_types.DataType] = None, targets: list[int] = None, average_over_targets = False, include_data_loss = False, include_unit_header_row = True):
    df, default_dq_type, targets_have = collect_data_quality(rec_dirs)
    if not dq_types:
        dq_types = [default_dq_type]
    if not targets:
        targets = targets_have
    summarize_and_store_data_quality(df, output_file_or_dir, dq_types, targets, average_over_targets, include_data_loss, include_unit_header_row)