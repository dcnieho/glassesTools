import numpy as np
import cv2
import pathlib

from . import data_files, ocv


class Pose:
    # description of tsv file used for storage
    _columns_compressed = {'frame_idx':1,
                           'pose_N_markers': 1, 'pose_R_vec': 3, 'pose_T_vec': 3,
                           'homography_N_markers': 1, 'homography_mat': 9}
    _non_float          = {'frame_idx': int, 'pose_ok': bool, 'pose_N_markers': int, 'homography_N_markers': int}

    def __init__(self,
                 frame_idx              : int,
                 pose_N_markers         : int       = 0,
                 pose_R_vec             : np.ndarray= None,
                 pose_T_vec             : np.ndarray= None,
                 homography_N_markers   : int       = 0,
                 homography_mat         : np.ndarray= None):
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
        self._iH          = None

    @staticmethod
    def readFromFile(fileName:str|pathlib.Path, start:int=None, end:int=None) -> dict[int,'Pose']:
        return data_files.read_file(fileName,
                                    Pose, True, True, False,
                                    start=start, end=end)[0]

    @staticmethod
    def writeToFile(poses: list['Pose'], fileName:str|pathlib.Path, skip_failed=False):
        data_files.write_array_to_file(poses, fileName,
                                       Pose._columns_compressed,
                                       skip_all_nan=skip_failed)

    def camToWorld(self, point: np.ndarray):
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(point)):
            return np.full((3,), np.nan)

        if self._RtMatInv is None:
            if self._RMatInv is None:
                if self._RMat is None:
                    self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
                self._RMatInv = self._RMat.T
            self._RtMatInv = np.hstack((self._RMatInv,np.matmul(-self._RMatInv,self.pose_T_vec.reshape(3,1))))

        return np.matmul(self._RtMatInv,np.append(np.array(point),1.).reshape((4,1))).flatten()

    def worldToCam(self, point: np.ndarray):
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(point)):
            return np.full((3,), np.nan)

        if self._RtMat is None:
            if self._RMat is None:
                self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
            self._RtMat = np.hstack((self._RMat, self.pose_T_vec.reshape(3,1)))

        return np.matmul(self._RtMat,np.append(np.array(point),1.).reshape((4,1))).flatten()

    def planeToCamHomography(self, point: np.ndarray, camera_params: ocv.CameraParams) -> np.ndarray:
        # from location on plane (2D) to location on camera image (2D)
        from . import transforms
        if self.homography_mat is None:
            return np.full((2,), np.nan)

        if self._iH is None:
            self._iH = np.linalg.inv(self.homography_mat)
        out = transforms.applyHomography(self._iH, *point)
        if camera_params.has_intrinsics():
            out = transforms.distortPoint(*out, camera_params.camera_mtx, camera_params.distort_coeffs)
        return out

    def camToPlaneHomography(self, point: np.ndarray, camera_params: ocv.CameraParams) -> np.ndarray:
        # from location on camera image (2D) to location on plane (2D)
        from . import transforms
        if self.homography_mat is None:
            return np.full((2,), np.nan)

        if camera_params.has_intrinsics():
            point = transforms.undistortPoint(*point, camera_params.camera_mtx, camera_params.distort_coeffs)
        return transforms.applyHomography(self.homography_mat, *point)

    def getOriginOnImage(self, camera_params: ocv.CameraParams) -> np.ndarray:
        if self.pose_N_markers>0 and camera_params.has_intrinsics():
            a = cv2.projectPoints(np.zeros((1,3)), self.pose_R_vec,self.pose_T_vec, camera_params.camera_mtx,camera_params.distort_coeffs)[0].flatten()
        elif self.homography_N_markers>0:
            a = self.planeToCamHomography([0., 0.], camera_params.camera_mtx, camera_params.distort_coeffs)
        else:
            a = np.full((2,), np.nan)
        return a

    def vectorIntersect(self, vector: np.ndarray, origin = np.array([0.,0.,0.])):
        from . import transforms

        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(vector)):
            return np.full((3,), np.nan)

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