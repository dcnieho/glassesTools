import numpy as np
import pandas as pd
import cv2
import math
from collections import defaultdict

from . import data_files


class Pose:
    _columns_compressed = {'frame_idx':1,
                           'pose_N_markers': 1, 'pose_R_vec': 3, 'pose_T_vec': 3,
                           'homography_N_markers': 1, 'homography_mat': 9}
    _non_float          = {'frame_idx': int, 'pose_ok': bool, 'pose_N_markers': int, 'homography_N_markers': int}

    def __init__(self,
                 frame_idx:int,
                 pose_N_markers=0,
                 pose_R_vec:np.ndarray=None,
                 pose_T_vec:np.ndarray=None,
                 homography_N_markers=0,
                 homography_mat:np.ndarray=None):
        self.frame_idx            : int         = frame_idx
        # pose
        self.pose_N_markers       : int         = pose_N_markers        # number of ArUco markers this pose estimate is based on. 0 if failed
        self.pose_R_vec           : np.ndarray  = pose_R_vec
        self.pose_T_vec           : np.ndarray  = pose_T_vec
        # homography
        self.homography_N_markers : int         = homography_N_markers  # number of ArUco markers this homongraphy estimate is based on. 0 if failed
        self.homography_mat       : np.ndarray  = homography_mat.reshape(3,3) if homography_mat is not None else homography_mat

        # internals
        self._RMat        = None
        self._RtMat       = None
        self._planeNormal = None
        self._planePoint  = None
        self._RMatInv     = None
        self._RtMatInv    = None

    @staticmethod
    def readFromFile(fileName,start=None,end=None) -> dict[int,'Pose']:
        readSubset  = start is not None and end is not None
        df          = pd.read_csv(str(fileName), delimiter='\t', index_col=False, dtype=defaultdict(lambda: float, **Pose._non_float))
        if readSubset:
            df = df[(df['frame_idx'] >= start) & (df['frame_idx'] <= end)]

        # figure out what the data columns are
        # group columns into numpy arrays, insert None if missing
        cols = [col for col in Pose._columns_compressed if Pose._columns_compressed[col]>1]
        allCols = tuple([c for c in df.columns if col in c] for col in cols)

        # drop rows where are all data columns are nan
        df = df.dropna(how='all',subset=[c for cs in allCols for c in cs])

        # group columns into numpy arrays, insert None if missing
        for c,ac in zip(cols,allCols):
            if ac:
                df[c] = [data_files.noneIfAnyNan(x) for x in df[ac].values]  # make list of numpy arrays, or None if there are any NaNs in the array
            else:
                df[c] = None

        # keep only the columns we want (this also puts them in the right order even if that doesn't matter since we use kwargs to construct objects)
        df = df[Pose._columns_compressed.keys()]

        poses = {idx:Pose(**kwargs) for idx,kwargs in zip(df['frame_idx'].values,df.to_dict(orient='records'))}
        return poses

    @staticmethod
    def writeToFile(poses: list['Pose'], fileName, skip_failed=False):
        data_files.write_array_to_file(poses, fileName,
                                       Pose._columns_compressed,
                                       skip_all_nan=skip_failed)

    def camToWorld(self, point):
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(point)):
            return np.array([np.nan, np.nan, np.nan])

        if self._RtMatInv is None:
            if self._RMatInv is None:
                if self._RMat is None:
                    self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
                self._RMatInv = self._RMat.T
            self._RtMatInv = np.hstack((self._RMatInv,np.matmul(-self._RMatInv,self.pose_T_vec.reshape(3,1))))

        return np.matmul(self._RtMatInv,np.append(np.array(point),1.).reshape((4,1))).flatten()

    def worldToCam(self, point):
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(point)):
            return np.array([np.nan, np.nan, np.nan])

        if self._RtMat is None:
            if self._RMat is None:
                self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
            self._RtMat = np.hstack((self._RMat, self.pose_T_vec.reshape(3,1)))

        return np.matmul(self._RtMat,np.append(np.array(point),1.).reshape((4,1))).flatten()

    def vectorIntersect(self, vector, origin = np.array([0.,0.,0.])):
        from . import transforms

        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(vector)):
            return np.array([np.nan, np.nan, np.nan])

        if self._planeNormal is None:
            if self._RtMat is None:
                if self._RMat is None:
                    self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
                self._RtMat = np.hstack((self._RMat, self.pose_T_vec.reshape(3,1)))

            # get poster normal
            self._planeNormal = np.matmul(self._RMat, np.array([0., 0., 1.]))
            # get point on poster (just use origin)
            self._planePoint  = np.matmul(self._RtMat, np.array([0., 0., 0., 1.]))

        # normalize vector
        vector /= np.sqrt((vector**2).sum())

        # find intersection of 3D gaze with poster
        return transforms.intersect_plane_ray(self._planeNormal, self._planePoint, vector.flatten(), origin.flatten())