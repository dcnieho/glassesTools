import numpy as np
import pandas as pd
import cv2
import pathlib
import bisect


class CameraParams:
    def __init__(self,
                 # camera info
                 resolution: np.ndarray,
                 # intrinsics
                 camera_mtx: np.ndarray, distort_coeffs: np.ndarray = None,
                 # extrinsics
                 rotation_vec: np.ndarray = None, position: np.ndarray = None):

        self.resolution     : np.ndarray = resolution
        self.camera_mtx     : np.ndarray = camera_mtx
        self.distort_coeffs : np.ndarray = distort_coeffs
        self.rotation_vec   : np.ndarray = rotation_vec
        self.position       : np.ndarray = position

    @staticmethod
    def readFromFile(fileName: str|pathlib.Path) -> 'CameraParams':
        fileName = pathlib.Path(fileName)
        if not fileName.is_file():
            return CameraParams(None,None)

        fs = cv2.FileStorage(fileName, cv2.FILE_STORAGE_READ)
        resolution      = fs.getNode("resolution").mat()
        cameraMatrix    = fs.getNode("cameraMatrix").mat()
        distCoeff       = fs.getNode("distCoeff").mat()
        # camera extrinsics for 3D gaze
        cameraRotation  = fs.getNode("rotation").mat()
        if cameraRotation is not None:
            cameraRotation  = cv2.Rodrigues(cameraRotation)[0]  # need rotation vector, not rotation matrix
        cameraPosition  = fs.getNode("position").mat()
        fs.release()

        return CameraParams(resolution,cameraMatrix,distCoeff,cameraRotation,cameraPosition)

    def has_intrinsics(self):
        return (self.camera_mtx is not None) and (self.distort_coeffs is not None)


class CV2VideoReader:
    def __init__(self, file: str|pathlib.Path, timestamps: list|np.ndarray|pd.DataFrame):
        self.file = pathlib.Path(file)
        if isinstance(timestamps,list):
            self.ts = np.array(timestamps)
        elif isinstance(timestamps, pd.DataFrame):
            self.ts = timestamps['timestamp'].to_numpy()
        else:
            self.ts = timestamps

        self.cap = cv2.VideoCapture(self.file)
        if not self.cap.isOpened():
            raise RuntimeError('the file "{}" could not be opened'.format(str(self.file)))
        self.nframes= float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = -1
        self._last_good_ts = (-1, -1., -1.)  # frame_idx, ts from opencv, ts from file
        self._is_off_by_one = False

    def __del__(self):
        self.cap.release()

    def get_prop(self, cv2_prop):
        return self.cap.get(cv2_prop)

    def set_prop(self, cv2_prop, val):
        return self.cap.set(cv2_prop, val)

    def read_frame(self, report_gap=False):
        ts0 = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = self.cap.read()
        ts1 = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.frame_idx += 1

        # check if this is a stream for which opencv returns timestamps that are one frame off
        if self.frame_idx==0 and ts0==ts1:
            self._is_off_by_one = True

        # check if we're done. Can't trust ret==False to indicate we're at end of video, as
        # it may also return False for some corrupted frames that we can just read past
        if not ret and (self.frame_idx==0 or self.frame_idx/self.nframes>.99):
            return True, None, None, None

        # keep going
        ts_from_list = self.ts[self.frame_idx]
        if self.frame_idx==1 or ts1>0.:
            # check for gap, and if there is a gap, fix up frame_idx if needed
            if self._last_good_ts[0]!=-1 and ts_from_list-self._last_good_ts[2] < ts1-self._last_good_ts[1]-1:  # little bit of leeway for precision or mismatched timestamps
                # we skipped some frames altogether, need to correct current frame_idx
                t_jump = ts1-self._last_good_ts[1]
                tss = self.ts-self._last_good_ts[2]
                # find best matching frame idx so we catch up with the jump
                idx = bisect.bisect(tss, t_jump)
                if abs(tss[idx-1]-t_jump)<abs(tss[idx]-t_jump):
                    idx -= 1
                self.frame_idx = idx
                ts_from_list = self.ts[self.frame_idx]
                if report_gap and self.frame_idx-self._last_good_ts[0]>1:
                    print(f'Frame discontinuity detected (jumped from {self._last_good_ts[0]} to {self.frame_idx}), there are probably corrupt frames in your video')
            self._last_good_ts = (self.frame_idx, ts1, ts_from_list)

        # we might not have a valid frame, but we're not done yet
        if not ret or frame is None:
            return False, None,  self.frame_idx, ts_from_list
        else:
            return False, frame, self.frame_idx, ts_from_list

    def report_frame(self, interval=100):
        if self.frame_idx%interval==0:
            print('  frame {}'.format(self.frame_idx))