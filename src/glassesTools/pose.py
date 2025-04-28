import pathlib
import cv2
import numpy as np
import typing
import enum

from glassesTools import annotation, data_files, drawing, intervals, marker, ocv, timestamps, transforms, _has_GUI
if _has_GUI:
    from .gui import video_player
else:
    # stub out video_player as a class so type annotations below do not fail
    class video_player:
        @property
        def GUI(self) -> typing.Any: ...

class Pose:
    # description of tsv file used for storage
    _columns_compressed = {'frame_idx':1,
                           'pose_N_markers': 1, 'pose_reprojection_error': 1, 'pose_R_vec': 3, 'pose_T_vec': 3,
                           'homography_N_markers': 1, 'homography_mat': 9}
    _non_float          = {'frame_idx': int, 'pose_ok': bool, 'pose_N_markers': int, 'homography_N_markers': int}

    def __init__(self,
                 frame_idx              : int,
                 pose_N_markers         : int       = 0,
                 pose_reprojection_error: float     = -1.,
                 pose_R_vec             : np.ndarray= None,
                 pose_T_vec             : np.ndarray= None,
                 homography_N_markers   : int       = 0,
                 homography_mat         : np.ndarray= None):
        self.frame_idx              : int         = frame_idx
        # pose
        self.pose_N_markers         : int         = pose_N_markers        # number of ArUco markers this pose estimate is based on. 0 if failed
        self.pose_reprojection_error: float       = pose_reprojection_error
        self.pose_R_vec             : np.ndarray  = pose_R_vec
        self.pose_T_vec             : np.ndarray  = pose_T_vec
        # homography
        self.homography_N_markers   : int         = homography_N_markers  # number of ArUco markers this homongraphy estimate is based on. 0 if failed
        self.homography_mat         : np.ndarray  = homography_mat.reshape(3,3) if homography_mat is not None else homography_mat

        # internals
        self._RMat              = None
        self._RtMat             = None
        self._plane_normal      = None
        self._plane_point       = None
        self._RMatInv           = None
        self._RtMatInv          = None
        self._i_homography_mat  = None

    def pose_successful(self):
        return self.pose_N_markers>0
    def homography_successful(self):
        return self.homography_N_markers>0

    def draw_frame_axis(self, img, camera_params: ocv.CameraParams, arm_length, thickness, sub_pixel_fac, position = [0.,0.,0.]):
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or not camera_params.has_intrinsics():
            return
        drawing.openCVFrameAxis(img, camera_params, self.pose_R_vec, self.pose_T_vec, arm_length, thickness, sub_pixel_fac, position)

    def cam_frame_to_world(self, point: np.ndarray):
        # NB: world frame is in the plane's coordinate system
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(point)):
            return np.full((3,), np.nan)

        if self._RtMatInv is None:
            if self._RMatInv is None:
                if self._RMat is None:
                    self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
                self._RMatInv = self._RMat.T
            self._RtMatInv = np.hstack((self._RMatInv,np.matmul(-self._RMatInv,self.pose_T_vec.reshape(3,1))))

        return np.matmul(self._RtMatInv,np.append(np.array(point),1.).reshape((4,1))).flatten()

    def world_frame_to_cam(self, point: np.ndarray):
        # NB: world frame is in the plane's coordinate system
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(point)):
            return np.full((3,), np.nan)

        if self._RtMat is None:
            if self._RMat is None:
                self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
            self._RtMat = np.hstack((self._RMat, self.pose_T_vec.reshape(3,1)))

        return np.matmul(self._RtMat,np.append(np.array(point),1.).reshape((4,1))).flatten()

    def plane_to_cam_pose(self, point_plane: np.ndarray, camera_params: ocv.CameraParams) -> np.ndarray:
        # from location on plane (2D) to location on camera image (2D)
        if (self.pose_R_vec is None) or (self.pose_T_vec is None):
            return np.full((2,), np.nan)
        return transforms.project_points(point_plane, camera_params, rot_vec=self.pose_R_vec, trans_vec=self.pose_T_vec).flatten()

    def cam_to_plane_pose(self, point: np.ndarray, camera_params: ocv.CameraParams) -> tuple[np.ndarray,np.ndarray]:
        # from location on camera image (2D) to location on plane (2D)
        # NB: also returns intermediate result (intersection with plane in camera space)
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(point)) or not camera_params.has_intrinsics():
            return np.full((2,), np.nan), np.full((3,), np.nan)

        g3D = transforms.unproject_points(point, camera_params)

        # find intersection of 3D gaze with plane
        pos_cam = self.vector_intersect(g3D)  # default vec origin (0,0,0) because we use g3D from camera's view point

        # above intersection is in camera space, turn into world space to get position on plane
        (x,y,z) = self.cam_frame_to_world(pos_cam) # z should be very close to zero
        return np.asarray([x, y]), pos_cam

    def plane_to_cam_homography(self, point: np.ndarray, camera_params: ocv.CameraParams) -> np.ndarray:
        # from location on plane (2D) to location on camera image (2D)
        if self.homography_mat is None:
            return np.full((2,), np.nan)

        if self._i_homography_mat is None:
            self._i_homography_mat = np.linalg.inv(self.homography_mat)
        out = transforms.apply_homography(point, self._i_homography_mat).flatten()
        if camera_params.has_intrinsics():
            out = transforms.distort_points(out, camera_params).flatten()
        return out

    def cam_to_plane_homography(self, point_cam: np.ndarray, camera_params: ocv.CameraParams) -> np.ndarray:
        # from location on camera image (2D) to location on plane (2D)
        if self.homography_mat is None:
            return np.full((2,), np.nan)

        if camera_params.has_intrinsics():
            point_cam = transforms.undistort_points(point_cam, camera_params).flatten()
        return transforms.apply_homography(point_cam, self.homography_mat).flatten()

    def get_origin_on_image(self, camera_params: ocv.CameraParams) -> np.ndarray:
        if self.pose_successful() and camera_params.has_intrinsics():
            a = self.plane_to_cam_pose(np.zeros((1,3)), camera_params)
        elif self.homography_successful():
            a = self.plane_to_cam_homography(np.zeros((1,2)), camera_params)
        else:
            a = np.full((2,), np.nan)
        return a

    def vector_intersect(self, vector: np.ndarray, origin = np.array([0.,0.,0.])):
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(vector)):
            return np.full((3,), np.nan)

        if self._plane_normal is None:
            if self._RtMat is None:
                if self._RMat is None:
                    self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
                self._RtMat = np.hstack((self._RMat, self.pose_T_vec.reshape(3,1)))

            # get poster normal
            self._plane_normal = self._RMat[:,2]     # equivalent to: np.matmul(self._RMat, np.array([0., 0., 1.]))
            # get point on poster (just use origin)
            self._plane_point  = self._RtMat[:,3]    # equivalent to: np.matmul(self._RtMat, np.array([0., 0., 0., 1.]))

        # normalize vector
        vector /= np.linalg.norm(vector)

        # find intersection of 3D gaze with poster
        return transforms.intersect_plane_ray(self._plane_normal, self._plane_point, vector.flatten(), origin.flatten())


def read_dict_from_file(fileName:str|pathlib.Path, episodes:list[list[int]]=None) -> dict[int,Pose]:
    return data_files.read_file(fileName,
                                Pose, True, True, False, False,
                                episodes=episodes)[0]

def write_list_to_file(poses: list[Pose], fileName:str|pathlib.Path, skip_failed=False):
    data_files.write_array_to_file(poses, fileName,
                                    Pose._columns_compressed,
                                    skip_all_nan=skip_failed)

class Status(enum.Enum):
    Ok = enum.auto()
    Skip = enum.auto()
    Finished = enum.auto()

_T  = typing.TypeVar("_T")

class Estimator:
    def __init__(self, video_file: str|pathlib.Path, frame_timestamp_file: str|pathlib.Path|timestamps.VideoTimestamps, camera_calibration_file: str|pathlib.Path|ocv.CameraParams):
        self.video_ts   = frame_timestamp_file if isinstance(frame_timestamp_file,timestamps.VideoTimestamps) else timestamps.VideoTimestamps(frame_timestamp_file)
        self.video      = ocv.CV2VideoReader(video_file, self.video_ts.timestamps)
        self.cam_params = camera_calibration_file if isinstance(camera_calibration_file,ocv.CameraParams) else ocv.CameraParams.read_from_file(camera_calibration_file)

        self.plane_functions    : dict[str, typing.Callable[[str,int,np.ndarray,ocv.CameraParams], tuple[np.ndarray,np.ndarray]|None]] = {}
        self.plane_intervals    : dict[str, list[int]|list[list[int]]]                                  = {}
        self.plane_visualizers  : dict[str, typing.Callable[[str,int,np.ndarray,np.ndarray], None]|None]= {}

        self.individual_marker_functions    : dict[_T, typing.Callable[[_T,int,np.ndarray,ocv.CameraParams], tuple[np.ndarray,np.ndarray|None|None]]] = {}
        self.individual_marker_intervals    : dict[_T, list[int]|list[list[int]]]                                   = {}
        self.individual_marker_visualizers  : dict[_T, typing.Callable[[_T,int,np.ndarray,np.ndarray], None]|None]  = {}

        self.extra_proc_functions   : dict[str, typing.Callable[[str,int,np.ndarray,ocv.CameraParams,typing.Any], typing.Any]]   = {}
        self.extra_proc_intervals   : dict[str, list[int]|list[list[int]]]                                  = {}
        self.extra_proc_parameters  : dict[str, dict[str,typing.Any]]                                       = {}
        self.extra_proc_visualizers : dict[str, typing.Callable[[str,int,np.ndarray,typing.Any], None]|None]= {}

        self._cache: tuple[Status, dict[str, Pose], dict[_T, marker.Pose], dict[str, list[int, typing.Any]], tuple[np.ndarray, int, float]] = None  # self._cache[4][1] is frame number

        self.gui                    : video_player.GUI          = None
        self.has_gui                                            = False
        self.allow_early_exit                                   = True
        self.progress_updater       : typing.Callable[[], None] = None

        self.do_visualize                                       = False
        self.sub_pixel_fac                                      = 8
        self.plane_axis_arm_length                              = 25
        self.individual_marker_axis_arm_length                  = 25
        self.show_extra_processing_output                       = True

        self._first_frame                                       = True

    def __del__(self):
        if self.has_gui:
            self.gui.stop()

    def add_plane(self, plane: str,
                  plane_function: typing.Callable[[str,int,np.ndarray,ocv.CameraParams], tuple[np.ndarray,np.ndarray]|None],
                  processing_intervals: list[int]|list[list[int]]=None,
                  plane_visualizer: typing.Callable[[str,int,np.ndarray,np.ndarray], None]=None):
        if not self._first_frame:
            raise RuntimeError(f'You cannot register planes once video processing has started')
        if plane in self.plane_functions:
            raise ValueError(f'Cannot register the plane "{plane}", it is already registered')
        self.plane_functions[plane]     = plane_function
        self.plane_intervals[plane]     = processing_intervals
        self.plane_visualizers[plane]   = plane_visualizer

    def add_individual_marker(self, key: _T,
                              individual_marker_function: typing.Callable[[_T,int,np.ndarray,ocv.CameraParams], tuple[np.ndarray,np.ndarray|None|None]],
                              processing_intervals: list[int]|list[list[int]]=None,
                              individual_marker_visualizer: typing.Callable[[str,int,np.ndarray,np.ndarray], None]=None):
        if not self._first_frame:
            raise RuntimeError(f'You cannot register individual markers once video processing has started')
        if key in self.individual_marker_functions:
            raise ValueError(f'Cannot register the individual marker {key}, it is already registered')
        self.individual_marker_functions[key]   = individual_marker_function
        self.individual_marker_intervals[key]   = processing_intervals
        self.individual_marker_visualizers[key] = individual_marker_visualizer

    def register_extra_processing_fun(self,
                                      name: str,
                                      func: typing.Callable[[str,int,np.ndarray,ocv.CameraParams,typing.Any], typing.Any],
                                      processing_intervals: list[int]|list[list[int]],
                                      func_parameters: dict[str],
                                      visualizer: typing.Callable[[str,int,np.ndarray,typing.Any], None]):
        if not self._first_frame:
            raise RuntimeError(f'You cannot register extra processing functions once video processing has started')
        if name in self.extra_proc_functions:
            raise ValueError(f'Cannot register the extra processing function "{name}", it is already registered')
        self.extra_proc_functions[name] = func
        self.extra_proc_intervals[name] = processing_intervals
        self.extra_proc_parameters[name]= func_parameters
        self.extra_proc_visualizers[name]= visualizer

    def attach_gui(self, gui: video_player.GUI, episodes: dict[annotation.Event, list[int]] = None, window_id: int = None):
        self.gui            = gui
        self.has_gui        = self.gui is not None
        self.do_visualize   = self.has_gui

        if self.has_gui:
            self.gui.set_show_timeline(True, self.video_ts, episodes, window_id)

    def set_allow_early_exit(self, allow_early_exit: bool):
        # if False, processing will not stop because last frame with a defined plane or extra processing is reached
        self.allow_early_exit = allow_early_exit

    def set_progress_updater(self, progress_updater: typing.Callable[[], None]):
        self.progress_updater = progress_updater

    def set_visualize_on_frame(self, do_visualize: bool):
        self.do_visualize = do_visualize

    def get_video_info(self) -> tuple[int, int, float]:
        return int(self.video.get_prop(cv2.CAP_PROP_FRAME_WIDTH)), \
               int(self.video.get_prop(cv2.CAP_PROP_FRAME_HEIGHT)), \
                   self.video.get_prop(cv2.CAP_PROP_FPS)

    def estimate_pose(self, object_points, img_points) -> tuple[int, np.ndarray, np.ndarray, float]:
        # NB: N_markers also flags success of the pose estimation. Also 0 if not successful or not possible (missing intrinsics)
        N_markers, R_vec, T_vec, reprojection_error = 0, None, None, -1.
        if object_points is None or not self.cam_params.has_intrinsics():
            return N_markers, R_vec, T_vec, reprojection_error

        if self.cam_params.has_opencv_camera():
            n_solutions, R_vec, T_vec, reprojection_error = \
                cv2.solvePnPGeneric(object_points, img_points, self.cam_params.camera_mtx, self.cam_params.distort_coeffs, np.empty(1), np.empty(1))
            if n_solutions:
                N_markers = int(object_points.shape[0]/4)
            reprojection_error = reprojection_error[0][0]
        else:
            # we have a camera not supported by OpenCV
            # undistort points and project to a identity camera space, so we can use opencv functionality
            points_w  = transforms.unproject_points(img_points, self.cam_params)
            points_cam= transforms.project_points(points_w, ocv.CameraParams(self.cam_params.resolution, np.identity(3), np.zeros((5,1))))
            n_solutions, R_vec, T_vec, _ = cv2.solvePnPGeneric(object_points, points_cam.reshape((-1,1,2)), np.identity(3), np.zeros((5,1)), np.empty(1), np.empty(1))
            # need to compute reprojection error ourselves, output of solvePnPGeneric is meaningless due to arbitrary camera point units
            if n_solutions:
                proj_points = transforms.project_points(object_points,self.cam_params, rot_vec=R_vec[0], trans_vec=T_vec[0])
                reprojection_error = cv2.norm(proj_points.astype('float32').reshape((-1,1,2)),img_points,cv2.NORM_L2) / np.sqrt(2*proj_points.shape[0])
            else:
                reprojection_error = np.nan
        if n_solutions:
            N_markers = int(object_points.shape[0]/4)
        return N_markers, R_vec[0], T_vec[0], reprojection_error

    def estimate_homography(self, object_points, img_points) -> tuple[np.ndarray, bool]:
        # NB: N_markers also flags success of the pose estimation. Also 0 if not successful or not possible (missing intrinsics)
        N_markers, H = 0, None
        if object_points is None:
            return N_markers, H

        # use undistorted marker corners if possible
        if self.cam_params.has_intrinsics():
            img_points = transforms.undistort_points(img_points.reshape((-1,2)),self.cam_params).reshape((-1,1,2))

        H = transforms.estimate_homography(object_points, img_points)
        if H is not None:
            N_markers = int(object_points.shape[0]/4)
        return N_markers, H

    # higher level functions for detecting + pose estimation
    def estimate_pose_and_homography(self, frame_idx, img_points, object_points) -> tuple[Pose, dict[str]]:
        pose = Pose(frame_idx)
        if img_points is not None and object_points is not None:
            # get camera pose
            pose.pose_N_markers, pose.pose_R_vec, pose.pose_T_vec, pose.pose_reprojection_error = \
                self.estimate_pose(object_points, img_points)

            # also get homography (direct image plane to plane in world transform)
            pose.homography_N_markers, pose.homography_mat = \
                self.estimate_homography(object_points, img_points)
        return pose

    def process_one_frame(self, wanted_frame_idx:int = None) -> tuple[Status, dict[str, Pose], dict[str, marker.Pose], dict[str, list[int, typing.Any]], tuple[np.ndarray, int, float]]:
        if self._first_frame and self.has_gui:
            self.gui.set_playing(True)

        if wanted_frame_idx is not None and self._cache is not None and self._cache[4][1]==wanted_frame_idx:
            return self._cache

        should_exit, frame, frame_idx, frame_ts = self.video.read_frame(report_gap=True, wanted_frame_idx=wanted_frame_idx)

        if should_exit or (self.allow_early_exit and \
            (
                (not self.plane_intervals             or intervals.beyond_last_interval(frame_idx, self.plane_intervals)) and \
                (not self.individual_marker_intervals or intervals.beyond_last_interval(frame_idx, self.individual_marker_intervals)) and \
                (not self.extra_proc_intervals        or intervals.beyond_last_interval(frame_idx, self.extra_proc_intervals))
            )):
            self._cache = Status.Finished, None, None, None, (None, None, None)
            return self._cache
        if self.progress_updater:
            self.progress_updater()

        if self.has_gui:
            if self._first_frame and frame is not None:
                self.gui.set_frame_size(frame.shape)
                self._first_frame = False

            requests = self.gui.get_requests()
            for r,_ in requests:
                if r=='exit':   # only requests we need to handle
                    self._cache = Status.Finished, None, None, None, (None, None, None)
                    return self._cache
                if r=='close':
                    self.has_gui = False
                    self.gui.stop()

        # check we're in a current interval, else skip processing
        # NB: have to spool through like this, setting specific frame to read
        # with cap.set(cv2.CAP_PROP_POS_FRAMES) doesn't seem to work reliably
        # for VFR video files
        planes_for_this_frame = [p for p in self.plane_functions if intervals.is_in_interval(frame_idx, self.plane_intervals[p])]
        indiv_markers_for_this_frame = [i for i in self.individual_marker_functions if intervals.is_in_interval(frame_idx, self.individual_marker_intervals[i])]
        extra_processing_for_this_frame = [e for e in self.extra_proc_functions if intervals.is_in_interval(frame_idx, self.extra_proc_intervals[e])]
        if frame is None or (not planes_for_this_frame and not indiv_markers_for_this_frame and not extra_processing_for_this_frame):
            # we don't have a valid frame or nothing to do, continue to next
            if self.has_gui:
                # do update timeline of the viewers
                self.gui.update_image(None, frame_ts/1000., frame_idx)
            self._cache = Status.Skip, None, None, None, (frame, frame_idx, frame_ts)
            return self._cache

        pose_out                : dict[str, Pose]                   = {}
        individual_marker_out   : dict[_T , marker.Pose]            = {}
        extra_processing_out    : dict[str, list[int, typing.Any]]  = {}
        if planes_for_this_frame:
            # detect fiducials
            plane_points: dict[str, tuple[np.ndarray,np.ndarray]] = {}
            for p in planes_for_this_frame:
                det_output = self.plane_functions[p](p, frame_idx, frame, self.cam_params)
                if det_output[0] is not None:
                    plane_points[p] = det_output
            # determine pose
            for p in plane_points:
                pose_out[p] = self.estimate_pose_and_homography(frame_idx, *plane_points[p])

        if indiv_markers_for_this_frame:
            # detect fiducials
            indiv_marker_points: dict[_T, tuple[np.ndarray,np.ndarray]] = {}
            for i in indiv_markers_for_this_frame:
                det_output = self.individual_marker_functions[i](i, frame_idx, frame, self.cam_params)
                if det_output[0] is not None:
                    indiv_marker_points[i] = det_output
            # determine pose, if wanted
            for i in indiv_marker_points:
                mpose = marker.Pose(frame_idx)
                if indiv_marker_points[i][1] is not None:   # object points may not be available (e.g. when marker size is not set)
                    # can only get marker pose if we have a calibrated camera (need intrinsics), else at least flag that marker was found
                    if self.cam_params.has_opencv_camera():
                        _, mpose.R_vec, mpose.T_vec = cv2.solvePnP(indiv_marker_points[i][1], indiv_marker_points[i][0], self.cam_params.camera_mtx, self.cam_params.distort_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    elif self.cam_params.has_colmap_camera():
                        # we have a camera not supported by OpenCV
                        # undistort points and project to a identity camera space, so we can use opencv functionality
                        points_w  = transforms.unproject_points(indiv_marker_points[i][0],self.cam_params)
                        points_cam= transforms.project_points(points_w, ocv.CameraParams(self.cam_params.resolution, np.identity(3), np.zeros((5,1))))
                        cv2.solvePnP(indiv_marker_points[i][1], points_cam, np.identity(3), np.zeros((5,1)), flags=cv2.SOLVEPNP_IPPE_SQUARE)
                individual_marker_out[i] = mpose

        for e in extra_processing_for_this_frame:
            extra_processing_out[e] = [frame_idx, *self.extra_proc_functions[e](e, frame_idx, frame, self.cam_params, **self.extra_proc_parameters[e])]

        # now that all processing is done, handle visualization, if any
        if self.do_visualize:
            # first draw all detection output
            if planes_for_this_frame:
                for p in plane_points:
                    if self.plane_visualizers[p] is None:
                        continue
                    self.plane_visualizers[p](p, frame_idx, frame, plane_points[p][0])
            if indiv_markers_for_this_frame:
                for i in indiv_marker_points:
                    if self.individual_marker_visualizers[i] is None:
                        continue
                    self.individual_marker_visualizers[i](i, frame_idx, frame, indiv_marker_points[i][0])
            for e in extra_processing_for_this_frame:
                if self.show_extra_processing_output and self.extra_proc_visualizers[e]:
                    self.extra_proc_visualizers[e](e, frame, *extra_processing_out[e])

            # now also draw pose, if wanted
            if self.plane_axis_arm_length:
                for p in pose_out:
                    if pose_out[p].pose_successful():
                        pose_out[p].draw_frame_axis(frame, self.cam_params, self.plane_axis_arm_length, 3, sub_pixel_fac=self.sub_pixel_fac)
            if self.individual_marker_axis_arm_length:
                for i in individual_marker_out:
                    if individual_marker_out[i].pose_successful():
                        individual_marker_out[i].draw_frame_axis(frame, self.cam_params, self.individual_marker_axis_arm_length, self.sub_pixel_fac)

        if self.has_gui:
            self.gui.update_image(frame, frame_ts/1000., frame_idx)

        self._cache = Status.Ok, pose_out, individual_marker_out, extra_processing_out, (frame, frame_idx, frame_ts)
        return self._cache

    def process_video(self) -> tuple[dict[str, list[Pose]], dict[_T, list[marker.Pose]], dict[str, list[list[int, typing.Any]]]]:
        poses_out               : dict[str, list[Pose]]                 = {p:[] for p in self.plane_functions}
        individual_markers_out  : dict[_T, list[marker.Pose]]           = {i:[] for i in self.individual_marker_functions}
        extra_processing_out    : dict[str, list[list[int, typing.Any]]]= {e:[] for e in self.extra_proc_functions}
        while True:
            status, plane, individual_marker, extra_proc, _ = self.process_one_frame()
            if status==Status.Finished:
                break
            if status==Status.Skip:
                continue
            # store outputs
            for p in plane:
                poses_out[p].append(plane[p])
            for i in individual_marker:
                individual_markers_out[i].append(individual_marker[i])
            for e in extra_proc:
                extra_processing_out[e].append(extra_proc[e])

        return poses_out, individual_markers_out, extra_processing_out