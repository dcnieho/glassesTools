import numpy as np
import pathlib

from . import data_files, drawing, ocv


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

    def draw_frame_axis(self, frame, camera_params: ocv.CameraParams, arm_length, sub_pixel_fac = 8):
        if not camera_params.has_intrinsics():
            return
        drawing.openCVFrameAxis(frame, camera_params.camera_mtx, camera_params.distort_coeffs, self.R_vec, self.T_vec, arm_length, 3, sub_pixel_fac)


def read_dict_from_file(fileName:str|pathlib.Path, episodes:list[list[int]]=None) -> dict[int,Pose]:
    return data_files.read_file(fileName,
                                Pose, True, True, False, False,
                                episodes=episodes)[0]

def write_list_to_file(poses: list[Pose], fileName:str|pathlib.Path, skip_failed=False):
    data_files.write_array_to_file(poses, fileName,
                                    Pose._columns_compressed,
                                    skip_all_nan=skip_failed)