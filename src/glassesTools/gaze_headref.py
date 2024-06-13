import numpy as np
import cv2
import pathlib

from . import data_files, drawing, ocv


class Gaze:
    # description of tsv file used for storage
    _columns_compressed = {'timestamp': 1, 'timestamp_VOR': 1, 'timestamp_ref': 1, 'frame_idx':1, 'frame_idx_VOR':1, 'frame_idx_ref':1,
                           'gaze_pos_vid': 2, 'gaze_pos_3d': 3, 'gaze_dir_l': 3, 'gaze_ori_l': 3, 'gaze_dir_r': 3, 'gaze_ori_r': 3}
    _non_float          = {'frame_idx': int, 'frame_idx_VOR': int, 'frame_idx_ref': int}
    _columns_optional   = ['timestamp_VOR', 'frame_idx_VOR', 'timestamp_ref', 'frame_idx_ref']

    def __init__(self,
                 timestamp      : float,
                 frame_idx      : int,

                 gaze_pos_vid   : np.ndarray,

                 timestamp_ori  : float      = None,
                 frame_idx_ori  : int        = None,
                 timestamp_VOR  : float      = None,
                 frame_idx_VOR  : int        = None,
                 timestamp_ref  : float      = None,
                 frame_idx_ref  : int        = None,

                 gaze_pos_3d    : np.ndarray = None,
                 gaze_dir_l     : np.ndarray = None,
                 gaze_ori_l     : np.ndarray = None,
                 gaze_dir_r     : np.ndarray = None,
                 gaze_ori_r     : np.ndarray = None):
        self.timestamp    : float       = timestamp
        self.frame_idx    : int         = frame_idx

        # optional timestamps and frame_idxs
        self.timestamp_ori: float       = timestamp_ori     # timestamp field can be set to any of these three. Keep here a copy of the original timestamp
        self.frame_idx_ori: int         = frame_idx_ori
        self.timestamp_VOR: float       = timestamp_VOR
        self.frame_idx_VOR: int         = frame_idx_VOR
        self.timestamp_ref: float       = timestamp_ref
        self.frame_idx_ref: int         = frame_idx_ref     # frameidx _in reference video_ not in this eye tracker's video (unless this is the reference)

        self.gaze_pos_vid : np.ndarray  = gaze_pos_vid      # gaze point on the scene video
        self.gaze_pos_3d  : np.ndarray  = gaze_pos_3d       # gaze point in the world (often binocular gaze point)
        self.gaze_dir_l   : np.ndarray  = gaze_dir_l
        self.gaze_ori_l   : np.ndarray  = gaze_ori_l
        self.gaze_dir_r   : np.ndarray  = gaze_dir_r
        self.gaze_ori_r   : np.ndarray  = gaze_ori_r

    def draw(self, img:np.ndarray, cameraParams:ocv.CameraParams=None, subPixelFac=1):
        drawing.openCVCircle(img, self.gaze_pos_vid, 8, (0,255,0), 2, subPixelFac)
        # draw 3D gaze point as well, usually coincides with 2D gaze point, but not always. E.g. the Adhawk MindLink may
        # apply a correction for parallax error to the projected gaze point using the vergence signal.
        if self.gaze_pos_3d is not None and cameraParams.has_intrinsics():
            camRot = np.zeros((1,3)) if cameraParams.rotation_vec is None else cameraParams.rotation_vec
            camPos = np.zeros((1,3)) if cameraParams.position     is None else cameraParams.position
            a = cv2.projectPoints(np.array(self.gaze_pos_3d).reshape(1,3),camRot,camPos,cameraParams.camera_mtx,cameraParams.distort_coeffs)[0][0][0]
            drawing.openCVCircle(img, a, 5, (0,255,255), -1, subPixelFac)


def read_dict_from_file(fileName:str|pathlib.Path, episodes:list[list[int]]=None, ts_column_suffixes: list[str] = None) -> tuple[dict[int,list[Gaze]], int]:
    return data_files.read_file(fileName,
                                Gaze, False, False, True, True,
                                episodes=episodes, ts_fridx_field_suffixes=ts_column_suffixes)

def write_dict_to_file(gazes: list[Gaze] | dict[int,list[Gaze]], fileName:str|pathlib.Path, skip_missing=False):
    data_files.write_array_to_file(gazes, fileName,
                                   Gaze._columns_compressed,
                                   Gaze._columns_optional,
                                   skip_all_nan=skip_missing)