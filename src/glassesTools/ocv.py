import numpy as np
import pandas as pd
import cv2
import pathlib
import bisect
import warnings


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
    def read_from_file(fileName: str|pathlib.Path) -> 'CameraParams':
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
        self.nframes = len(self.ts)
        self.frame_idx = -1
        self._last_good_ts = (-1, -1., -1.)  # frame_idx, ts from opencv, ts from file
        self._cache: tuple[bool, np.ndarray, int, float] = None # self._cache[2] is frame index

    def __del__(self):
        self.cap.release()

    def get_prop(self, cv2_prop):
        return self.cap.get(cv2_prop)

    def set_prop(self, cv2_prop, val):
        return self.cap.set(cv2_prop, val)

    # NB: we seek by spooling, because I found seeking through setting cv2.CAP_PROP_POS_MSEC unreliable
    def read_frame(self, report_gap=False, wanted_frame_idx:int=None) -> tuple[bool, np.ndarray, int, float]:
        if wanted_frame_idx!=None:
            assert wanted_frame_idx>=0 and wanted_frame_idx<self.nframes, f'wanted_frame_idx ({wanted_frame_idx}) out of bounds ([0-{self.nframes-1}])'
        else:
            wanted_frame_idx = self.frame_idx+1

        if self.frame_idx>wanted_frame_idx:
            warnings.warn(f'Requested frame ({wanted_frame_idx}) was earlier than current position of reader (frame {self.frame_idx}). Impossible to deliver because this video reader strictly advances forward. Returning last read frame', RuntimeWarning)
            # this condition can only occur if we've already read something and thus have a cache, so this assert should never trigger
            assert self._cache is not None, f'No cache, unexpected failure mode, contact developer'
            return self._cache
        elif self._cache is not None and self._cache[2]==wanted_frame_idx:
            return self._cache

        while True:
            ret, frame = self.cap.read()
            # get timestamp of this frame according to OpenCV
            # NB: this timestamp does not take into account edit lists in an mp4
            # it seems (compare with output of .\ffprobe.exe file.mp4 -select_streams 0 -show_entries frame=pts_time)
            # which is one of the reasons we provide our own timestamps
            ocv_ts = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.frame_idx += 1

            # check if we're done. Can't trust ret==False to indicate we're at end of video, as
            # it may also return False for some corrupted frames that we can just read past
            if not ret and (self.frame_idx==0 or self.frame_idx/self.nframes>.99):
                self._cache = True, None, None, None
                return self._cache

            # keep going
            ts_from_list = self.ts[self.frame_idx]
            if self.frame_idx==1 or ocv_ts>0.:
                # check for gap, and if there is a gap, fix up frame_idx if needed
                if self._last_good_ts[0]!=-1 and ts_from_list-self._last_good_ts[2] < ocv_ts-self._last_good_ts[1]-1:  # little bit of leeway (1ms) for precision or mismatched timestamps
                    # we skipped some frames altogether, need to correct current frame_idx
                    t_jump = ocv_ts-self._last_good_ts[1]   # compare OpenCV timestamps to get size of jump
                    tss = self.ts-self._last_good_ts[2]     # apply jump to our own timestamps (so, we're robust to e.g. OpenCV ignoring the edit list)
                    # find best matching frame idx so we catch up with the jump
                    self.frame_idx = self._find_closest_idx(t_jump, tss)
                    ts_from_list = self.ts[self.frame_idx]
                    if report_gap and self.frame_idx-self._last_good_ts[0]>1:
                        print(f'Frame discontinuity detected (jumped from {self._last_good_ts[0]} to {self.frame_idx}), there are probably corrupt frames in your video')
                self._last_good_ts = (self.frame_idx, ocv_ts, ts_from_list)

            # keep spooling until we arrive at the wanted frame
            if self.frame_idx==wanted_frame_idx:
                if not ret or frame is None:
                    # we might not have a valid frame, but we're not done yet
                    self._cache = False, None,  self.frame_idx, ts_from_list
                else:
                    self._cache = False, frame, self.frame_idx, ts_from_list
                return self._cache

    def _find_closest_idx(self, time: float, times: np.ndarray) -> int:
        idx = bisect.bisect(times, time)
        if abs(times[idx-1]-time)<abs(times[idx]-time):
            idx -= 1
        return idx

    def report_frame(self, interval=100):
        if self.frame_idx%interval==0:
            print('  frame {}'.format(self.frame_idx))