import numpy as np
import pandas as pd
import cv2
import pathlib
import bisect
import warnings
import pycolmap
import copy
from typing import Any


class CameraParams:
    def __init__(self,
                 # camera info
                 resolution: np.ndarray,
                 # intrinsics
                 camera_mtx: np.ndarray, distort_coeffs: np.ndarray = None,
                 # extrinsics
                 rotation_vec: np.ndarray = None, position: np.ndarray = None,
                 # colmap camera dict
                 colmap_camera_dict: dict[str,Any] = None):

        # info about camera
        self.resolution     : np.ndarray = resolution.flatten() if resolution is not None else None
        # intrinsics
        self.camera_mtx     : np.ndarray = camera_mtx
        self.distort_coeffs : np.ndarray = distort_coeffs.flatten() if distort_coeffs is not None else None
        # extrinsics
        self.rotation_vec   : np.ndarray = rotation_vec.flatten() if rotation_vec is not None else None
        self.position       : np.ndarray = position.flatten() if position is not None else None

        # colmap, for more extensive camera models
        self.colmap_camera: pycolmap.Camera = None
        self.colmap_camera_no_distortion: pycolmap.Camera = None

        # initialize the colmap cameras
        if colmap_camera_dict:
            self.colmap_camera = pycolmap.Camera(colmap_camera_dict)
        elif self.has_opencv_camera():
            # turn into colmap camera
            self.colmap_camera = pycolmap.Camera.create(0, pycolmap.CameraModelId.FULL_OPENCV, self.camera_mtx[0,0], *self.resolution)

            cal_params = np.zeros((self.colmap_camera.extra_params_idxs()[-1]+1,))
            cal_params[self.colmap_camera.focal_length_idxs()]     = [self.camera_mtx[0,0], self.camera_mtx[1,1]]
            cal_params[self.colmap_camera.principal_point_idxs()]  = self.camera_mtx[0:2,2]
            if len(self.distort_coeffs)>len(self.colmap_camera.extra_params_idxs()):
                self.colmap_camera = None
                print(f'Warning: could not make colmap FULL_OPENCV camera as there are too many distortion parameters {len(self.distort_coeffs)}')
            else:
                cal_params[self.colmap_camera.extra_params_idxs()[0:len(self.distort_coeffs)]] = self.distort_coeffs.flatten()
                self.colmap_camera.params = cal_params
        if self.colmap_camera is not None:
            # make version with distortion parameters zeroed out
            cam_dict = copy.deepcopy(self.colmap_camera.todict())
            cam_dict['params'][4:] = 0
            self.colmap_camera_no_distortion = pycolmap.Camera(cam_dict)


    @staticmethod
    def read_from_file(fileName: str|pathlib.Path) -> 'CameraParams':
        fileName = pathlib.Path(fileName)
        if not fileName.is_file():
            return CameraParams(None,None)

        fs = cv2.FileStorage(fileName, cv2.FILE_STORAGE_READ)
        resolution      = fs.getNode("resolution").mat()
        # intrinsics
        cameraMatrix    = fs.getNode("cameraMatrix").mat()
        distCoeff       = fs.getNode("distCoeff").mat()
        # extrinsics
        cameraRotation  = fs.getNode("rotation").mat()
        if cameraRotation is not None:
            cameraRotation  = cv2.Rodrigues(cameraRotation)[0]  # need rotation vector, not rotation matrix
        cameraPosition  = fs.getNode("position").mat()
        # colmap camera dict, if available
        colmap_camera = {}
        colmap_camera_n = fs.getNode("colmap_camera")
        if not colmap_camera_n.empty():
            colmap_camera['camera_id'] = int(colmap_camera_n.getNode('camera_id').real())
            colmap_camera['model'] = getattr(pycolmap.CameraModelId, colmap_camera_n.getNode('model').string())
            colmap_camera['width'] = int(colmap_camera_n.getNode('width').real())
            colmap_camera['height'] = int(colmap_camera_n.getNode('height').real())
            colmap_camera['params'] = colmap_camera_n.getNode("params").mat()
            colmap_camera['has_prior_focal_length'] = bool(colmap_camera_n.getNode('width').real())
        fs.release()

        return CameraParams(resolution, cameraMatrix,distCoeff, cameraRotation,cameraPosition, colmap_camera)

    def has_opencv_camera(self):
        return (self.camera_mtx is not None) and (self.distort_coeffs is not None)
    def has_colmap_camera(self):
        return self.colmap_camera is not None
    def has_intrinsics(self):
        return self.has_opencv_camera() or self.has_colmap_camera()
    def has_extrinsics(self):
        return (self.rotation_vec is not None) and (self.position is not None)


class CV2VideoReader:
    def __init__(self, file: str|pathlib.Path, timestamps: list|np.ndarray|pd.DataFrame):
        self.file = pathlib.Path(file)
        if isinstance(timestamps,list):
            self._ts = np.array(timestamps)
        elif isinstance(timestamps, pd.DataFrame):
            self._ts = timestamps['timestamp'].to_numpy()
        else:
            self._ts = timestamps

        self._cap = cv2.VideoCapture(self.file)
        if not self._cap.isOpened():
            raise RuntimeError('the file "{}" could not be opened'.format(str(self.file)))
        self.nframes = len(self._ts)
        self.frame_idx = -1
        self._last_good_ts = (-1, -1., -1.)  # frame_idx, ts from opencv, ts from file
        self._cache: tuple[bool, np.ndarray, int, float, dict[str,Any]] = None # self._cache[2] is frame index

        # check if there is a file with info about each frame (such as size, and offset on the sensor if an ROI is used) and if so, read it and check it matches the number of frames in the video
        frame_info_file = self.file.parent / (self.file.stem+'_frame_info.tsv')
        if frame_info_file.is_file():
            # read frame info from file
            self.frame_info = pd.read_csv(frame_info_file, sep='\t', index_col='frame_idx')
            if len(self.frame_info)!=self.nframes:
                raise ValueError(f'The number of frames in the frame info file ({len(self.frame_info)}) does not match the number of frames in the video ({self.nframes}). Please check your files.')
        else:
            self.frame_info = None


    def __del__(self):
        self._cap.release()

    def get_prop(self, cv2_prop):
        return self._cap.get(cv2_prop)

    def set_prop(self, cv2_prop, val):
        return self._cap.set(cv2_prop, val)

    def has_offsets(self):
        return self.frame_info is not None and 'offset_x' in self.frame_info.columns and 'offset_y' in self.frame_info.columns

    def check_cam_params(self, cam_params: CameraParams):
        if cam_params.resolution is None:
            return
        vid_height = int(self.get_prop(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_width = int(self.get_prop(cv2.CAP_PROP_FRAME_WIDTH))
        if cam_params.resolution[0]!=vid_width or cam_params.resolution[1]!=vid_height:
            if self.has_offsets():
                # check the size of the frame in the frame info matches the video
                fi_widths = self.frame_info['width'].unique()
                fi_heights = self.frame_info['height'].unique()
                if len(fi_widths)!=1 or len(fi_heights)!=1:
                    raise ValueError(f"The frame info file has more than one frame width or height, which means that the recording resolution of the video changed during the recording. This should not be possible. Please check your files.")
                if fi_widths[0]==vid_width and fi_heights[0]==vid_height:
                    # everything consistent -> ok
                    return
                raise ValueError(f"The resolution of the video matches neither that set in the camera parameters ({cam_params.resolution[0]}x{cam_params.resolution[1]}), nor that expected from the frame info file ({fi_widths[0]}x{fi_heights[0]}). The video has resolution {vid_width}x{vid_height}. Please check your files.")
            raise ValueError(f"The resolution of the video does not match that set in the camera parameters ({cam_params.resolution[0]}x{cam_params.resolution[1]}). The video has resolution {vid_width}x{vid_height}. In this situation, a frame info file should be provided (expected name {self.file.stem+'_frame_info.tsv'}) containing info about where the ROI was on the camera sensor. This file was not found or did not contain the expected information (columns 'offset_x', 'offset_y', 'width' and 'height'). Please check your files.")

    # NB: we seek by spooling, because I found seeking through setting cv2.CAP_PROP_POS_MSEC unreliable
    def read_frame(self, report_gap=False, wanted_frame_idx:int=None) -> tuple[bool, np.ndarray, int, float, dict[str,Any]]:
        if wanted_frame_idx!=None:
            if wanted_frame_idx<0 or wanted_frame_idx>=self.nframes:
                raise ValueError(f'wanted_frame_idx ({wanted_frame_idx}) out of bounds ([0-{self.nframes-1}])')
        else:
            wanted_frame_idx = self.frame_idx+1

        if self.frame_idx>wanted_frame_idx:
            warnings.warn(f'Requested frame ({wanted_frame_idx}) was earlier than current position of reader (frame {self.frame_idx}). Impossible to deliver because this video reader strictly advances forward. Returning last read frame', RuntimeWarning)
            # this condition can only occur if we've already read something and thus have a cache, so this check should never trigger
            if self._cache is None:
                raise RuntimeError(f'No cache, unexpected failure mode, contact developer')
            return self._cache
        elif self._cache is not None and self._cache[2]==wanted_frame_idx:
            return self._cache

        while True:
            ret, frame = self._cap.read()
            # get timestamp of this frame according to OpenCV
            # NB: this timestamp does not take into account edit lists in an mp4
            # it seems (compare with output of .\ffprobe.exe file.mp4 -select_streams 0 -show_entries frame=pts_time)
            # which is one of the reasons we provide our own timestamps
            ocv_ts = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            self.frame_idx += 1

            # check if we're done. Can't trust ret==False to indicate we're at end of video, as
            # it may also return False for some corrupted frames that we can just read past
            if not ret and (self.frame_idx==0 or self.frame_idx/self.nframes>.99):
                self._cache = True, None, None, None, {}
                return self._cache

            # keep going
            ts_from_list = self._ts[self.frame_idx]
            if self.frame_idx==1 or ocv_ts>0.:
                # check for gap, and if there is a gap, fix up frame_idx if needed
                if self._last_good_ts[0]!=-1 and ts_from_list-self._last_good_ts[2] < ocv_ts-self._last_good_ts[1]-1:  # little bit of leeway (1ms) for precision or mismatched timestamps
                    # we skipped some frames altogether, need to correct current frame_idx
                    t_jump = ocv_ts-self._last_good_ts[1]   # compare OpenCV timestamps to get size of jump
                    tss = self._ts-self._last_good_ts[2]     # apply jump to our own timestamps (so, we're robust to e.g. OpenCV ignoring the edit list)
                    # find best matching frame idx so we catch up with the jump
                    self.frame_idx = self._find_closest_idx(t_jump, tss)
                    ts_from_list = self._ts[self.frame_idx]
                    if report_gap and self.frame_idx-self._last_good_ts[0]>1:
                        print(f'Frame discontinuity detected (jumped from {self._last_good_ts[0]} to {self.frame_idx}), there are probably corrupt frames in your video')
                self._last_good_ts = (self.frame_idx, ocv_ts, ts_from_list)

            # keep spooling until we arrive at the wanted frame
            if self.frame_idx==wanted_frame_idx:
                if not ret or frame is None:
                    # we might not have a valid frame, but we're not done yet
                    self._cache = False, None,  self.frame_idx, ts_from_list, {}
                else:
                    self._cache = False, frame, self.frame_idx, ts_from_list, {} if self.frame_info is None or self.frame_idx not in self.frame_info.index else self.frame_info.loc[self.frame_idx].to_dict()
                return self._cache

    def _find_closest_idx(self, time: float, times: np.ndarray) -> int:
        idx = bisect.bisect(times, time)
        if abs(times[idx-1]-time)<abs(times[idx]-time):
            idx -= 1
        return idx

    def report_frame(self, interval=100):
        if self.frame_idx%interval==0:
            print('  frame {}'.format(self.frame_idx))