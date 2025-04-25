import numpy as np
import pandas as pd
import pathlib
import typing
from collections import defaultdict

from . import data_files, drawing, json, naming, ocv

class MarkerID(typing.NamedTuple):
    m_id:           int
    aruco_dict_id:  int
def marker_ID_to_str(m: MarkerID):
    from . import aruco
    return f'{m.m_id} ({aruco.dict_id_to_str[m.aruco_dict_id]})'
def _serialize_marker_id(m: MarkerID):
    from . import aruco
    return {'m_id':m.m_id, 'aruco_dict':aruco.dict_id_to_str[m.aruco_dict_id]}
def _deserialize_marker_id(m: dict[str,str|int]):
    from . import aruco
    return MarkerID(m_id = m['m_id'],
                    aruco_dict_id = aruco.str_to_dict_id(m['aruco_dict_id' if 'aruco_dict_id' in m else 'aruco_dict']))
json.register_type(json.TypeEntry(MarkerID, '__config.MarkerID__', _serialize_marker_id, _deserialize_marker_id))

class Marker:
    def __init__(self, key: int, center: np.ndarray, corners: list[np.ndarray]=None, color: str=None, rot: float=0.):
        self.key = key
        self.center = center
        self.corners = corners
        self.color = color
        self.rot = rot

    def __str__(self):
        ret = '[%d]: center @ (%.2f, %.2f), rot %.0f deg' % (self.key, self.center[0], self.center[1], self.rot)
        return ret

    def shift(self, offset=np.ndarray):
        self.center += offset
        if self.corners:
            for c in self.corners:
                c += offset

def corners_intersection(corners):
    line1 = ( corners[0], corners[2] )
    line2 = ( corners[1], corners[3] )
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array( [x,y] ).astype('float32')

class Pose:
    # description of tsv file used for storage
    _columns_compressed = {'frame_idx': 1, 'R_vec': 3, 'T_vec': 3}
    _non_float          = {'frame_idx': int}

    def __init__(self,
                 frame_idx  : int,
                 R_vec      : np.ndarray= None,
                 T_vec      : np.ndarray= None):
        self.frame_idx  : int         = frame_idx
        self.R_vec      : np.ndarray  = R_vec
        self.T_vec      : np.ndarray  = T_vec

    def pose_successful(self):
        return self.R_vec is not None and self.T_vec is not None

    def draw_frame_axis(self, frame, camera_params: ocv.CameraParams, arm_length, sub_pixel_fac = 8):
        if not camera_params.has_intrinsics():
            return
        drawing.openCVFrameAxis(frame, camera_params, self.R_vec, self.T_vec, arm_length, 3, sub_pixel_fac)


def read_dict_from_file(fileName:str|pathlib.Path, episodes:list[list[int]]=None) -> dict[int,Pose]:
    return data_files.read_file(fileName,
                                Pose, True, True, False, False,
                                episodes=episodes)[0]

def write_list_to_file(poses: list[Pose], fileName:str|pathlib.Path, skip_failed=False):
    data_files.write_array_to_file(poses, fileName,
                                   Pose._columns_compressed,
                                   skip_all_nan=skip_failed)

def get_file_name(marker_id: int, aruco_dict_id: int, folder: str|pathlib.Path|None) -> pathlib.Path:
    from . import aruco
    file_name = f'{naming.marker_pose_prefix}{aruco.dict_id_to_str[aruco_dict_id]}_{marker_id}.tsv'
    if folder is None:
        return file_name
    folder = pathlib.Path(folder)
    return folder / file_name

def read_dataframe_from_file(marker_id: int, aruco_dict_id: int, folder: str|pathlib.Path) -> pd.DataFrame:
    file = get_file_name(marker_id, aruco_dict_id, folder)
    return pd.read_csv(file,sep='\t', dtype=defaultdict(lambda: float, **Pose._non_float))


@typing.overload
def code_for_presence(markers: pd.DataFrame, allow_failed=False) -> pd.DataFrame: ...
def code_for_presence(markers: dict[typing.Any, pd.DataFrame], allow_failed=False) -> dict[typing.Any, pd.DataFrame]: ...
def code_for_presence(markers: pd.DataFrame|dict[typing.Any, pd.DataFrame], allow_failed=False) -> pd.DataFrame|dict[typing.Any, pd.DataFrame]:
    if isinstance(markers,dict):
        for i in markers:
            markers[i] = _code_for_presence_impl(markers[i], f'{i}_', allow_failed)
    else:
        markers = _code_for_presence_impl(markers,'', allow_failed)
    return markers

def _code_for_presence_impl(markers: pd.DataFrame, lbl_extra:str, allow_failed=False) -> pd.DataFrame:
    new_col_lbl = f'marker_{lbl_extra}presence'
    markers.insert(len(markers.columns),
        new_col_lbl,
        True if allow_failed else markers[[c for c in markers.columns if c not in ['frame_idx']]].notnull().all(axis='columns')
    )
    markers = markers[['frame_idx',new_col_lbl]] if 'frame_idx' in markers else markers[[new_col_lbl]]
    markers = markers.astype({new_col_lbl: bool}) # ensure the new column is bool
    return markers

def expand_detection(markers: pd.DataFrame, fill_value):
    if 'frame_idx' in markers.columns:
        min_fr_idx = markers['frame_idx'].min()
        max_fr_idx = markers['frame_idx'].max()
        new_index = pd.Index(range(min_fr_idx,max_fr_idx+1), name='frame_idx')
        return markers.set_index('frame_idx').reindex(new_index, fill_value=fill_value).reset_index()
    else:
        if markers.index.name!='frame_idx':
            raise ValueError(f'It was expected that the name of the index is "frame_idx". It was "{markers.index.name}" instead. This may mean this dataframe does not contain the expected information. Cannot continue.')
        min_fr_idx = markers.index.min()
        max_fr_idx = markers.index.max()
        new_index = pd.Index(range(min_fr_idx,max_fr_idx+1), name='frame_idx')
        return markers.reindex(new_index, fill_value=fill_value)

def get_appearance_starts_ends(m: pd.DataFrame, max_gap_duration: int, min_duration: int):
    vals   = np.pad(m['marker_presence'].values.astype(int), (1, 1), 'constant', constant_values=(0, 0))
    d      = np.diff(vals)
    starts = np.nonzero(d == 1)[0]
    ends   = np.nonzero(d == -1)[0]
    gaps   = starts[1:]-ends[:-1]
    # fill gaps in marker detection
    gapi   = np.nonzero(gaps<=max_gap_duration)[0]
    starts = np.delete(starts,gapi+1)
    ends   = np.delete(ends,gapi)
    # remove too short
    lengths= ends-starts
    shorti = np.nonzero(lengths<min_duration)[0]
    starts = np.delete(starts,shorti)
    ends   = np.delete(ends,shorti)
    # turn first and last frames into frame_idx values
    if 'frame_idx' in m.columns:
        return m.loc[starts,'frame_idx'].to_numpy(), m.loc[ends-1,'frame_idx'].to_numpy() # NB: -1 so that ends point to last frame during which marker was last seen (and to not index out of the array)
    elif m.index.name=='frame_idx':
        return m.index[starts].to_numpy(), m.index[ends-1].to_numpy()

def get_sequence_interval(starts: dict[int,list[int]], ends: dict[int,list[int]], pattern: list[int], max_intermarker_gap_duration: int, side='start') -> np.ndarray:
    # find marker pattern (sequence of markers following in right order with gap no longer than max_intermarker_gap_duration)
    pairs: list[tuple[int,int]] = []
    for i in range(len(ends[pattern[0]])):
        end_idx = i
        for j in range(len(pattern)-1):
            if end_idx is None:
                break
            end     = ends[pattern[j]][end_idx]
            gaps    = starts[pattern[j+1]]-end
            end_idx = get_smallest_gap_end(gaps,max_intermarker_gap_duration)
        if end_idx is not None:
            pairs.append((starts[pattern[0]][i], ends[pattern[-1]][end_idx]))

    idx = 0 if side=='start' else 1
    return np.array([p[idx] for p in pairs])

def get_smallest_gap_end(gaps: np.ndarray, max_intermarker_gap_duration: int):
    gapi = np.nonzero(np.logical_and(gaps>=0, gaps<=max_intermarker_gap_duration))[0]
    if gapi.size:
        # if there are multiple that qualify, take the smallest gap
        mini = np.argmin(gaps[gapi])
        return gapi[mini]
    return None
