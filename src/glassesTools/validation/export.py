import pathlib
import pandas as pd

from . import DataQualityType

def collect_data_quality(rec_dirs: list[str | pathlib.Path], file_name: str|dict[str,str]='dataQuality.tsv', col_for_parent=None):
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
    df = pd.concat((pd.read_csv(rec[0], delimiter='\t').assign(**rec[1]) for rec in rec_files), ignore_index=True)
    if df.empty:
        return None, None, None
    # set indices
    df = df.set_index(idx_vals+['marker_interval','type','target'])
    # change type index into enum
    typeIdx = df.index.names.index('type')
    df.index = df.index.set_levels(pd.CategoricalIndex([getattr(DataQualityType,x) for x in df.index.levels[typeIdx]]),level='type')

    # see what we have
    dq_types = sorted(list(df.index.levels[typeIdx]), key=lambda dq: dq.value)
    targets  = list(df.index.levels[df.index.names.index('target')])

    # good default selection of dq type to export
    if DataQualityType.pose_vidpos_ray in dq_types:
        default_dq_type = DataQualityType.pose_vidpos_ray
    elif DataQualityType.pose_vidpos_homography in dq_types:
        default_dq_type = DataQualityType.pose_vidpos_homography
    else:
        # ultimate fallback, just set first available as the one to export
        default_dq_type = dq_types[0]

    return df, default_dq_type, targets

def summarize_and_store_data_quality(df: pd.DataFrame, output_file_or_dir: str | pathlib.Path, dq_types: list[DataQualityType], targets: list[int], average_over_targets = False, include_data_loss = False):
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
    # remove unwanted data loss
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
    df.to_csv(output_file_or_dir, mode='w', header=True, sep='\t', na_rep='nan', float_format="%.6f")

def export_data_quality(rec_dirs: list[str | pathlib.Path], output_file_or_dir: str | pathlib.Path, dq_types: list[DataQualityType] = None, targets: list[int] = None, average_over_targets = False, include_data_loss = False):
    df, default_dq_type, targets_have = collect_data_quality(rec_dirs)
    if not dq_types:
        dq_types = [default_dq_type]
    if not targets:
        targets = targets_have
    summarize_and_store_data_quality(df, output_file_or_dir, dq_types, targets, average_over_targets, include_data_loss)

def export_et_sync(rec_dirs: list[str|pathlib.Path], in_file_name: str, output_file_or_dir: str|pathlib.Path):
    sync_files = [(pathlib.Path(rec)/in_file_name,{'recording': rec.name, 'session': rec.parent.name}) for rec in rec_dirs]
    sync_files = [f for f in sync_files if f[0].is_file()]
    # get all sync files
    df = pd.concat((pd.read_csv(sync[0], delimiter='\t').assign(**sync[1]) for sync in sync_files), ignore_index=True)
    if df.empty:
        return
    df = df.set_index(['session','recording','interval'])
    # store
    output_file_or_dir = pathlib.Path(output_file_or_dir)
    if output_file_or_dir.is_dir():
        output_file_or_dir = output_file_or_dir / 'et_sync.tsv'
    df.to_csv(output_file_or_dir, mode='w', header=True, sep='\t', na_rep='nan', float_format="%.6f")