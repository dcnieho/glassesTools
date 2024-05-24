import numpy as np
import pandas as pd
import cv2
from collections import defaultdict

from . import drawing


class Gaze:
    _columns_compressed = {'timestamp': 1, 'frame_idx':1,
                           'gaze_pos_vid': 2, 'gaze_pos_3d': 3, 'gaze_dir_l': 3, 'gaze_ori_l': 3, 'gaze_dir_r': 3, 'gaze_ori_r': 3}
    _non_float          = {'frame_idx': int}

    def __init__(self,
                 timestamp:float,
                 frame_idx:int,
                 gaze_pos_vid:np.ndarray,
                 gaze_pos_3d:np.ndarray=None,
                 gaze_dir_l:np.ndarray=None,
                 gaze_ori_l:np.ndarray=None,
                 gaze_dir_r:np.ndarray=None,
                 gaze_ori_r:np.ndarray=None):
        self.timestamp   : float        = timestamp
        self.frame_idx   : int          = frame_idx

        self.gaze_pos_vid: np.ndarray   = gaze_pos_vid      # gaze point on the scene video
        self.gaze_pos_3d : np.ndarray   = gaze_pos_3d       # gaze point in the world (often binocular gaze point)
        self.gaze_dir_l  : np.ndarray   = gaze_dir_l
        self.gaze_ori_l  : np.ndarray   = gaze_ori_l
        self.gaze_dir_r  : np.ndarray   = gaze_dir_r
        self.gaze_ori_r  : np.ndarray   = gaze_ori_r

    @staticmethod
    def readFromFile(fileName) -> dict[int,list['Gaze']]:
        df = pd.read_csv(str(fileName), delimiter='\t', index_col=False, dtype=defaultdict(lambda: float, **Gaze._non_float))

        # group columns into numpy arrays, insert None if missing
        cols = [col for col in Gaze._columns_compressed if Gaze._columns_compressed[col]>1]
        allCols = tuple([c for c in df.columns if col in c] for col in cols)
        for c,ac in zip(cols,allCols):
            if ac:
                df[c] = [x for x in df[ac].values]  # make list of numpy arrays
            else:
                df[c] = None

        # keep only the columns we want (this also puts them in the right order even if that doesn't matter since we use kwargs to construct objects)
        df = df[Gaze._columns_compressed.keys()]

        # make the gaze objects
        gaze = [Gaze(**kwargs) for kwargs in df.to_dict(orient='records')]

        # organize into dict by frame index
        gazes = {}
        for k,v in zip(df['frame_idx'].values,gaze):
            gazes.setdefault(k, []).append(v)

        return gazes, df['frame_idx'].max()

    def draw(self, img, subPixelFac=1, camRot=None, camPos=None, cameraMatrix=None, distCoeff=None):
        drawing.openCVCircle(img, self.gaze_pos_vid, 8, (0,255,0), 2, subPixelFac)
        # draw 3D gaze point as well, usually coincides with 2D gaze point, but not always. E.g. the Adhawk MindLink may
        # apply a correction for parallax error to the projected gaze point using the vergence signal.
        if self.gaze_pos_3d is not None and camRot is not None and camPos is not None and cameraMatrix is not None and distCoeff is not None:
            a = cv2.projectPoints(np.array(self.gaze_pos_3d).reshape(1,3),camRot,camPos,cameraMatrix,distCoeff)[0][0][0]
            drawing.openCVCircle(img, a, 5, (0,255,255), -1, subPixelFac)