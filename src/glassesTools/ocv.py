import numpy as np
import pandas as pd
import cv2
import pathlib
import bisect


def arucoRefineDetectedMarkers(detector, image, arucoBoard, detectedCorners, detectedIds, rejectedCorners, cameraMatrix = None, distCoeffs= None):
    corners, ids, rejectedImgPoints, recoveredIds = detector.refineDetectedMarkers(
                            image = image, board = arucoBoard,
                            detectedCorners = detectedCorners, detectedIds = detectedIds, rejectedCorners = rejectedCorners,
                            cameraMatrix = cameraMatrix, distCoeffs = distCoeffs)
    if corners and corners[0].shape[0]==4:
        # there are versions out there where there is a bug in output shape of each set of corners, fix up
        corners = [np.reshape(c,(1,4,2)) for c in corners]
    if rejectedImgPoints and rejectedImgPoints[0].shape[0]==4:
        # same as for corners
        rejectedImgPoints = [np.reshape(c,(1,4,2)) for c in rejectedImgPoints]

    return corners, ids, rejectedImgPoints, recoveredIds


def readCameraCalibrationFile(fileName):
    fs = cv2.FileStorage(str(fileName), cv2.FILE_STORAGE_READ)
    cameraMatrix    = fs.getNode("cameraMatrix").mat()
    distCoeff       = fs.getNode("distCoeff").mat()
    # camera extrinsics for 3D gaze
    cameraRotation  = fs.getNode("rotation").mat()
    if cameraRotation is not None:
        cameraRotation  = cv2.Rodrigues(cameraRotation)[0]  # need rotation vector, not rotation matrix
    cameraPosition  = fs.getNode("position").mat()
    fs.release()

    return (cameraMatrix,distCoeff,cameraRotation,cameraPosition)


class CV2VideoReader:
    def __init__(self, file: str|pathlib.Path, timestamps: list|np.ndarray|pd.DataFrame):
        self.file = pathlib.Path(file)
        if isinstance(timestamps,list):
            self.ts = np.array(timestamps)
        elif isinstance(timestamps, pd.DataFrame):
            self.ts = timestamps['timestamp'].to_numpy()
        else:
            self.ts = timestamps

        self.cap = cv2.VideoCapture(str(self.file))
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

    def read_frame(self):
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
            self._last_good_ts = (self.frame_idx, ts1, ts_from_list)

        # we might not have a valid frame, but we're not done yet
        if not ret or frame is None:
            return False, None,  self.frame_idx, ts_from_list
        else:
            return False, frame, self.frame_idx, ts_from_list