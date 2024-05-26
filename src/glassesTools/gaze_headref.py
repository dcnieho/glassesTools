import numpy as np
import cv2
import pathlib

from . import data_files, drawing, ocv


class Gaze:
    # description of tsv file used for storage
    _columns_compressed = {'timestamp': 1, 'frame_idx':1,
                           'gaze_pos_vid': 2, 'gaze_pos_3d': 3, 'gaze_dir_l': 3, 'gaze_ori_l': 3, 'gaze_dir_r': 3, 'gaze_ori_r': 3}
    _non_float          = {'frame_idx': int}

    def __init__(self,
                 timestamp      : float,
                 frame_idx      : int,
                 gaze_pos_vid   : np.ndarray,
                 gaze_pos_3d    : np.ndarray = None,
                 gaze_dir_l     : np.ndarray = None,
                 gaze_ori_l     : np.ndarray = None,
                 gaze_dir_r     : np.ndarray = None,
                 gaze_ori_r     : np.ndarray = None):
        self.timestamp   : float        = timestamp
        self.frame_idx   : int          = frame_idx

        self.gaze_pos_vid: np.ndarray   = gaze_pos_vid      # gaze point on the scene video
        self.gaze_pos_3d : np.ndarray   = gaze_pos_3d       # gaze point in the world (often binocular gaze point)
        self.gaze_dir_l  : np.ndarray   = gaze_dir_l
        self.gaze_ori_l  : np.ndarray   = gaze_ori_l
        self.gaze_dir_r  : np.ndarray   = gaze_dir_r
        self.gaze_ori_r  : np.ndarray   = gaze_ori_r

    @staticmethod
    def readFromFile(fileName:str|pathlib.Path) -> tuple[dict[int,list['Gaze']], int]:
        return data_files.read_file(fileName,
                                    Gaze, False, False, True)

    def draw(self, img:np.ndarray, cameraParams:ocv.CameraParams=None, subPixelFac=1):
        drawing.openCVCircle(img, self.gaze_pos_vid, 8, (0,255,0), 2, subPixelFac)
        # draw 3D gaze point as well, usually coincides with 2D gaze point, but not always. E.g. the Adhawk MindLink may
        # apply a correction for parallax error to the projected gaze point using the vergence signal.
        if self.gaze_pos_3d is not None and cameraParams.has_intrinsics():
            camRot = np.zeros((1,3)) if cameraParams.rotation_vec is None else cameraParams.rotation_vec
            camPos = np.zeros((1,3)) if cameraParams.position     is None else cameraParams.position
            a = cv2.projectPoints(np.array(self.gaze_pos_3d).reshape(1,3),camRot,camPos,cameraParams.camera_mtx,cameraParams.distort_coeffs)[0][0][0]
            drawing.openCVCircle(img, a, 5, (0,255,255), -1, subPixelFac)