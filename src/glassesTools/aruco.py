import numpy as np
import cv2
import pathlib
from typing import Any, Callable
from enum import Enum, auto

from . import annotation, drawing, intervals, marker, ocv, plane, timestamps
from .gui import video_player


def deploy_marker_images(output_dir: str|pathlib.Path, size: int, ArUco_dict: int=cv2.aruco.DICT_4X4_250, markerBorderBits: int=1):
    # Load the predefined dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(ArUco_dict)

    # Generate the markers
    for i in range(dictionary.bytesList.shape[0]):
        markerImage = np.zeros((size, size), dtype=np.uint8)
        markerImage = cv2.aruco.generateImageMarker(dictionary, i, size, markerImage, markerBorderBits)

        cv2.imwrite(output_dir / f"{i}.png", markerImage)

class ArUcoDetector():
    def __init__(self, ArUco_dict: cv2.aruco.Dictionary, params: dict[str]):
        self._det_params                          = cv2.aruco.DetectorParameters()
        self._det_params.cornerRefinementMethod   = cv2.aruco.CORNER_REFINE_SUBPIX    # good default, user can override
        for p in params:
            setattr(self._det_params, p, params[p])

        self._dict                      = ArUco_dict
        self._det                       = cv2.aruco.ArucoDetector(self._dict, self._det_params)
        self._board: cv2.aruco.Board    = None

        self._camera_params             = ocv.CameraParams(None, None)

    def create_board(self, board_corner_points, ids):
        self._board = create_board(board_corner_points, ids, self._dict)

    def set_board(self, board):
        self._board = board

    def set_intrinsics(self, camera_params: ocv.CameraParams):
        self._camera_params = camera_params

    def detect_markers(self, image: cv2.UMat, min_nmarker_refine = 3):
        corners, ids, rejected_corners = _detect_markers(image, self._det)
        recovered_ids = None

        # Refine detected markers (eliminates markers not part of our poster, adds missing markers to the poster)
        if self._board and ids is not None and min_nmarker_refine and len(ids)>min_nmarker_refine:
            corners, ids, rejected_corners, recovered_ids = \
                self.refine_detected_markers(image, corners, ids, rejected_corners)
        return corners, ids, rejected_corners, recovered_ids

    def refine_detected_markers(self, image, detected_corners, detected_ids, rejected_corners):
        return _refine_detection(image, detected_corners, detected_ids, rejected_corners, self._det, self._board, self._camera_params.camera_mtx, self._camera_params.distort_coeffs)

    def _match_image_points(self, corners, ids):
        return self._board.matchImagePoints(corners, ids) # -> objP, imgP

    def estimate_pose(self, corners, ids) -> tuple[int, np.ndarray, np.ndarray, float]:
        objP, imgP = self._match_image_points(corners, ids)
        return self._estimate_pose_impl(objP, imgP)

    def _estimate_pose_impl(self, objP, imgP):
        # NB: N_markers also flags success of the pose estimation. Also 0 if not successful or not possible (missing intrinsics)
        N_markers, R_vec, T_vec, reprojection_error = 0, None, None, -1.
        if objP is None or not self._camera_params.has_intrinsics():
            return N_markers, R_vec, T_vec, reprojection_error

        n_solutions, R_vec, T_vec, reprojection_error = \
            cv2.solvePnPGeneric(objP, imgP, self._camera_params.camera_mtx, self._camera_params.distort_coeffs, np.empty(1), np.empty(1))
        if n_solutions:
            N_markers = int(objP.shape[0]/4)
        return N_markers, R_vec[0], T_vec[0], reprojection_error[0][0]

    def estimate_homography(self, corners, ids) -> tuple[np.ndarray, bool]:
        objP, imgP = self._match_image_points(corners, ids)
        return self._estimate_homography_impl(objP, imgP)

    def _estimate_homography_impl(self, objP, imgP):
        # NB: N_markers also flags success of the pose estimation. Also 0 if not successful or not possible (missing intrinsics)
        from . import transforms
        N_markers, H = 0, None
        if objP is None:
            return N_markers, H

        # use undistorted marker corners if possible
        if self._camera_params.has_intrinsics():
            imgP = np.vstack([cv2.undistortPoints(x, self._camera_params.camera_mtx, self._camera_params.distort_coeffs, P=self._camera_params.camera_mtx) for x in imgP])

        H, status = transforms.estimate_homography(objP, imgP)
        if status:
            N_markers = int(objP.shape[0]/4)
        return N_markers, H

    # higher level functions for detecting + pose estimation, and for results visualization
    def estimate_pose_and_homography(self, frame_idx, min_num_markers, corners, ids) -> tuple[plane.Pose, dict[str]]:
        pose = plane.Pose(frame_idx)
        if ids is not None and len(ids) >= min_num_markers:
            # get matching image and board points
            objP, imgP = self._match_image_points(corners, ids)

            # get camera pose
            pose.pose_N_markers, pose.pose_R_vec, pose.pose_T_vec, pose.pose_reprojection_error = \
                self._estimate_pose_impl(objP, imgP)

            # also get homography (direct image plane to plane in world transform)
            pose.homography_N_markers, pose.homography_mat = \
                self._estimate_homography_impl(objP, imgP)
        return pose

    def detect_and_estimate(self, frame, frame_idx, min_num_markers) -> tuple[plane.Pose, dict[str]]:
        corners, ids, rejectedImgPoints, recoveredIds = self.detect_markers(frame, min_nmarker_refine=min_num_markers)
        pose = self.estimate_pose_and_homography(frame_idx, min_num_markers, corners, ids)
        return pose, {'corners': corners, 'ids': ids, 'rejectedImgPoints': rejectedImgPoints, 'recoveredIds': recoveredIds}

    def visualize(self, frame, pose: plane.Pose, detect_dict, arm_length, sub_pixel_fac = 8, show_detected_markers = True, show_board_axis = True, show_rejected_markers = False, special_highlight = None):
        if special_highlight is None:
            special_highlight = []
        # for debug, can draw rejected markers on frame
        if show_rejected_markers:
            cv2.aruco.drawDetectedMarkers(frame, detect_dict['rejectedImgPoints'], None, borderColor=(0,0,255))

        # if any markers were detected, draw where on the frame
        if show_detected_markers:
            special_highlight += [detect_dict['recoveredIds'],(255,255,0)]
            drawing.arucoDetectedMarkers(frame, detect_dict['corners'], detect_dict['ids'], sub_pixel_fac=sub_pixel_fac, special_highlight=special_highlight)

        if show_board_axis:
            if pose.pose_successful():
                # draw axis indicating plane pose (origin and orientation)
                drawing.openCVFrameAxis(frame, self._camera_params.camera_mtx, self._camera_params.distort_coeffs, pose.pose_R_vec, pose.pose_T_vec, arm_length, 3, sub_pixel_fac)

            if pose.homography_successful():
                # find where plane origin is expected to be in the image
                target = pose.plane_to_cam_homography([0., 0.], self._camera_params)
                # draw target location on image
                if target[0] >= 0 and target[0] < frame.shape[1] and target[1] >= 0 and target[1] < frame.shape[0]:
                    drawing.openCVCircle(frame, target, 3, (0,0,0), -1, sub_pixel_fac)




def _detect_markers(image: cv2.UMat, det: cv2.aruco.ArucoDetector):
    corners, ids, rejected_corners = det.detectMarkers(image)
    if np.any(ids==None):
        ids = None
    return corners, ids, rejected_corners

def _refine_detection(image: cv2.UMat, detected_corners, detected_ids, rejected_corners, det: cv2.aruco.ArucoDetector, board: cv2.aruco.Board, camera_mtx, distort_coeffs):
    corners, ids, rejectedImgPoints, recoveredIds = \
        det.refineDetectedMarkers(
            image = image, board = board,
            detectedCorners = detected_corners, detectedIds = detected_ids, rejectedCorners = rejected_corners,
            cameraMatrix = camera_mtx, distCoeffs = distort_coeffs
            )
    if corners and corners[0].shape[0]==4:
        # there are versions out there where there is a bug in output shape of each set of corners, fix up
        corners = [np.reshape(c,(1,4,2)) for c in corners]
    if rejectedImgPoints and rejectedImgPoints[0].shape[0]==4:
        # same as for corners
        rejectedImgPoints = [np.reshape(c,(1,4,2)) for c in rejectedImgPoints]

    return corners, ids, rejectedImgPoints, recoveredIds

def _refine_for_multiple_planes(image, detected_corners, detected_ids, rejected_corners, detectors: dict[str, ArUcoDetector], camera_mtx, distort_coeffs):
    result = {}
    for d in detectors:
        a,b,c,e = _refine_detection(image, detected_corners, detected_ids, rejected_corners, detectors[d]._det, detectors[d]._board, detectors[d]._camera_params.camera_mtx, detectors[d]._camera_params.distort_coeffs)
        result[d] = dict(zip(['corners', 'ids', 'rejectedImgPoints', 'recoveredIds'],(a,b,c,e)))
    return result

def create_board(board_corner_points: list[np.ndarray], ids: list[int], ArUco_dict: cv2.aruco.Dictionary):
    board_corner_points = np.dstack(board_corner_points)        # list of 2D arrays -> 3D array
    board_corner_points = np.rollaxis(board_corner_points,-1)   # 4x2xN -> Nx4x2
    board_corner_points = np.pad(board_corner_points,((0,0),(0,0),(0,1)),'constant', constant_values=(0.,0.)) # Nx4x2 -> Nx4x3
    return cv2.aruco.Board(board_corner_points, ArUco_dict, np.array(ids))

class Status(Enum):
    Ok = auto()
    Skip = auto()
    Finished = auto()

class PoseEstimator:
    def __init__(self, video_file: str|pathlib.Path, frame_timestamp_file: str|pathlib.Path|timestamps.VideoTimestamps, camera_calibration_file: str|pathlib.Path|ocv.CameraParams):
        self.video_ts   = frame_timestamp_file if isinstance(frame_timestamp_file,timestamps.VideoTimestamps) else timestamps.VideoTimestamps(frame_timestamp_file)
        self.video      = ocv.CV2VideoReader(video_file, self.video_ts.timestamps)
        self.cam_params = camera_calibration_file if isinstance(camera_calibration_file,ocv.CameraParams) else ocv.CameraParams.read_from_file(camera_calibration_file)

        self.planes                 : list[str]                             = []
        self.plane_setups           : dict[str, dict[str]]                  = {}
        self.plane_proc_intervals   : dict[str, list[int]|list[list[int]]]  = {}
        self._aruco_boards          : dict[str, cv2.aruco.Board]            = {}
        self._all_aruco_ids         : set[int]                              = set()
        self._detectors             : dict[str, ArUcoDetector]              = {}
        self._single_detect_pass                                            = False
        self.allow_early_exit                                               = True

        self.individual_markers                 : dict[int, dict[str]]  = {}
        self._individual_marker_object_points   : dict[int, np.ndarray] = {}
        self.proc_individial_markers_all_frames                         = False

        self.extra_proc_functions   : dict[str, Callable[[np.ndarray,Any], Any]]    = {}
        self.extra_proc_intervals   : dict[str, list[int]|list[list[int]]]          = {}
        self.extra_proc_parameters  : dict[str]                                     = {}
        self.extra_proc_visualizer  : dict[str, Callable[[np.ndarray,Any], None]]   = {}

        self.gui                    : video_player.GUI  = None
        self.has_gui                                    = False

        self.do_visualize                               = False
        self.sub_pixel_fac                              = 8
        self.show_detected_markers                      = True
        self.show_plane_axes                            = True
        self.show_individual_marker_axes                = True
        self.show_sync_func_output                      = True
        self.show_unexpected_markers                    = True
        self.show_rejected_markers                      = False

        self._first_frame       = True
        self._do_report_frames  = True

    def __del__(self):
        if self.has_gui:
            self.gui.stop()

    def add_plane(self, plane: str, planes_setup: dict[str], processing_intervals: list[int]|list[list[int]] = None):
        if plane in self.planes:
            raise ValueError(f'Cannot register the plane "{plane}", it is already registered')
        self.planes.append(plane)
        self.plane_setups[plane] = planes_setup
        self.plane_proc_intervals[plane] = processing_intervals

        self._aruco_boards[plane]   = planes_setup['plane'].get_aruco_board()
        self._all_aruco_ids.update(self._aruco_boards[plane].getIds())
        self._detectors[plane]      = ArUcoDetector(self._aruco_boards[plane].getDictionary(), planes_setup['aruco_params'])
        self._detectors[plane].set_board(self._aruco_boards[plane])
        self._detectors[plane].set_intrinsics(self.cam_params)

        self._cache: tuple[Status, dict[str, plane.Pose], dict[str, marker.Pose], dict[str, list[int, Any]], tuple[np.ndarray, int, float]] = None  # self._cache[4][1] is frame number

        # check if we can do an optimization of detecting the markers only once for multiple planes (if it makes sense because we have more than one plane)
        aruco_dicts = [self.plane_setups[p]['aruco_dict'] for p in self.planes if 'aruco_dict' in self.plane_setups[p]]
        self._single_detect_pass = len(self.planes)>1 and len(aruco_dicts)==len(self.planes) and len(set(aruco_dicts))==1 and all([self.plane_setups[self.planes[0]]['aruco_params']==self.plane_setups[p]['aruco_params'] for p in self.planes])

    def add_individual_marker(self, marker_id: int, marker_setup):
        if not self._single_detect_pass and len(self.planes)!=1:
            raise ValueError("Detecting and reporting individual markers is only supported when there is a single plane, or all planes have identical ArUco setup")
        if marker_id in self.individual_markers:
            raise ValueError(f'Cannot register the individual marker with id {marker_id}, it is already registered')
        self.individual_markers[marker_id] = marker_setup
        marker_size = self.individual_markers[marker_id]['marker_size']
        self._individual_marker_object_points[marker_id] = np.array([[-marker_size/2,  marker_size/2, 0],
                                                                     [ marker_size/2,  marker_size/2, 0],
                                                                     [ marker_size/2, -marker_size/2, 0],
                                                                     [-marker_size/2, -marker_size/2, 0]])
        self._all_aruco_ids.add(marker_id)

    def register_extra_processing_fun(self,
                                      name: str,
                                      func: Callable[[np.ndarray,Any], Any],
                                      processing_intervals: list[int]|list[list[int]],
                                      func_parameters: dict[str],
                                      visualizer: Callable[[np.ndarray, Any], None]):
        if name in self.extra_proc_functions:
            raise ValueError(f'Cannot register the extra processing function "{name}", it is already registered')
        self.extra_proc_functions[name] = func
        self.extra_proc_intervals[name] = processing_intervals
        self.extra_proc_parameters[name]= func_parameters
        self.extra_proc_visualizer[name]= visualizer

    def attach_gui(self, gui: video_player.GUI, episodes: dict[annotation.Event, list[int]] = None, window_id: int = None):
        self.gui                    = gui
        self.has_gui                = self.gui is not None
        self.do_visualize           = self.has_gui

        if self.has_gui:
            self.gui.set_show_timeline(True, self.video_ts, episodes, window_id)

    def set_allow_early_exit(self, allow_early_exit: bool):
        # if False, processing will not stop because last frame with a defined plane or extra processing is reached
        self.allow_early_exit = allow_early_exit

    def set_visualize_on_frame(self, do_visualize: bool):
        self.do_visualize           = do_visualize

    def set_do_report_frames(self, do_report_frames: bool):
        self._do_report_frames = do_report_frames

    def get_video_info(self) -> tuple[int, int, float]:
        return int(self.video.get_prop(cv2.CAP_PROP_FRAME_WIDTH)), \
               int(self.video.get_prop(cv2.CAP_PROP_FRAME_HEIGHT)), \
                   self.video.get_prop(cv2.CAP_PROP_FPS)

    def process_one_frame(self, wanted_frame_idx:int = None) -> tuple[Status, dict[str, plane.Pose], dict[str, marker.Pose], dict[str, list[int, Any]], tuple[np.ndarray, int, float]]:
        if self._first_frame and self.has_gui:
            self.gui.set_playing(True)

        if wanted_frame_idx is not None and self._cache is not None and self._cache[4][1]==wanted_frame_idx:
            return self._cache

        should_exit, frame, frame_idx, frame_ts = self.video.read_frame(report_gap=True, wanted_frame_idx=wanted_frame_idx)

        if should_exit or (self.allow_early_exit and not self.proc_individial_markers_all_frames and \
            (
                intervals.beyond_last_interval(frame_idx, self.plane_proc_intervals) and \
                (not self.extra_proc_intervals or intervals.beyond_last_interval(frame_idx, self.extra_proc_intervals))
            )):
            self._cache = Status.Finished, None, None, None, (None, None, None)
            return self._cache
        if self._do_report_frames:
            self.video.report_frame()

        if self.has_gui:
            if self._first_frame and frame is not None:
                self.gui.set_frame_size(frame.shape)
                self._first_frame = False

            requests = self.gui.get_requests()
            for r,_ in requests:
                if r=='exit':   # only request we need to handle
                    should_exit = True
                    self._cache = Status.Finished, None, None, None, (None, None, None)
                    return self._cache

        # check we're in a current interval, else skip processing
        # NB: have to spool through like this, setting specific frame to read
        # with cap.set(cv2.CAP_PROP_POS_FRAMES) doesn't seem to work reliably
        # for VFR video files
        planes_for_this_frame = [p for p in self.planes if intervals.is_in_interval(frame_idx, self.plane_proc_intervals[p])]
        extra_processing_for_this_frame = [e for e in self.extra_proc_functions if intervals.is_in_interval(frame_idx, self.extra_proc_intervals[e])]
        if frame is None or (not (self.proc_individial_markers_all_frames and self.individual_markers) and not planes_for_this_frame and not extra_processing_for_this_frame):
            # we don't have a valid frame or nothing to do, continue to next
            if self.has_gui:
                # do update timeline of the viewers
                self.gui.update_image(None, frame_ts/1000., frame_idx)
            self._cache = Status.Skip, None, None, None, (frame, frame_idx, frame_ts)
            return self._cache

        pose_out                : dict[str, plane.Pose]     = {}
        individual_marker_out   : dict[str, marker.Pose]    = {}
        extra_processing_out    : dict[str, list[int, Any]] = {}
        if planes_for_this_frame or (self.proc_individial_markers_all_frames and self.individual_markers):
            # detect markers
            detect_dicts = {}
            if self._single_detect_pass or self.individual_markers:
                corners, ids, rejected_corners = _detect_markers(frame, self._detectors[self.planes[0]]._det)
                if planes_for_this_frame:
                    detect_dicts = _refine_for_multiple_planes(frame, corners, ids, rejected_corners, {p:self._detectors[p] for p in planes_for_this_frame}, self.cam_params.camera_mtx, self.cam_params.distort_coeffs)
            else:
                for p in planes_for_this_frame:
                    detect_dicts[p] = dict(zip(['corners', 'ids', 'rejectedImgPoints', 'recoveredIds'], self._detectors[p].detect_markers(frame, self.plane_setups[p]['min_num_markers'])))
            # determine pose
            for p in planes_for_this_frame:
                pose = self._detectors[p].estimate_pose_and_homography(frame_idx, self.plane_setups[p]['min_num_markers'], detect_dicts[p]['corners'], detect_dicts[p]['ids'])
                pose_out[p] = pose

            # deal with individual markers, if any
            if self.individual_markers and ids is not None:
                found_markers = np.where([x[0] in self.individual_markers for x in ids])[0]
                for idx in found_markers:
                    m_id = ids[idx][0]
                    pose = marker.Pose(frame_idx)
                    if self.cam_params.has_intrinsics():
                        # can only get marker pose if we have a calibrated camera (need intrinsics), else at least flag that marker was found
                        _, pose.R_vec, pose.T_vec = cv2.solvePnP(self._individual_marker_object_points[m_id], corners[idx], self.cam_params.camera_mtx, self.cam_params.distort_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    individual_marker_out[m_id] = pose

        for e in extra_processing_for_this_frame:
            extra_processing_out[e] = [frame_idx, *self.extra_proc_functions[e](frame,**self.extra_proc_parameters[e])]

        # now that all processing is done, handle visualization, if any
        if self.do_visualize:
            # for visualization purposes, either filter detected markers so only the expected ones
            # are shown (self.show_unknown_markers==False), or show them in a different color (self.show_unknown_markers==True)
            to_highlight = set()
            # get expected markers
            expected = set()
            if planes_for_this_frame:
                for p in planes_for_this_frame:
                    expected.update(self._detectors[p]._board.getIds())
                for m_id in self.individual_markers:
                    expected.add(m_id)
            else:
                expected = self._all_aruco_ids
            # filter dicts to find unexpected markers
            for p in detect_dicts:
                if detect_dicts[p]['ids'] is None:
                    continue
                unexpected = set(detect_dicts[p]['ids'].flatten())-expected
                if not unexpected:
                    continue
                if self.show_unexpected_markers:
                    to_highlight |= unexpected
                else:
                    to_remove = np.where([x[0] in unexpected for x in detect_dicts[p]['ids']])[0]
                    detect_dicts[p]['ids'] = np.delete(detect_dicts[p]['ids'], to_remove, axis=0)
                    detect_dicts[p]['corners'] = tuple(v for i,v in enumerate(detect_dicts[p]['corners']) if i not in to_remove)
            special_highlights = None
            if to_highlight:
                special_highlights = [list(to_highlight), (150,253,253)]
            # now actually draw them
            for p in planes_for_this_frame:
                # draw detection and pose, if wanted
                self._detectors[p].visualize(frame, pose_out[p], detect_dicts[p], self.plane_setups[p]['plane'].marker_size/2, self.sub_pixel_fac, self.show_detected_markers, self.show_plane_axes, self.show_rejected_markers, special_highlights)

            # ensure visualization of detected and rejected markers is honored when marker detection
            # was done but self._detectors[p].visualize() above won't be called because there are no
            # planes for this frame
            if not planes_for_this_frame and (self.proc_individial_markers_all_frames or self.individual_markers):
                if self.show_rejected_markers:
                    cv2.aruco.drawDetectedMarkers(frame, rejected_corners, None, special=(0,0,255))
                if self.show_detected_markers:
                    drawing.arucoDetectedMarkers(frame, corners, ids, sub_pixel_fac=self.sub_pixel_fac)

            # deal with individual markers, if any
            for m_id in individual_marker_out:
                if self.show_detected_markers:
                    # draw the detected marker in a different color
                    idx = np.where(ids==m_id)[0][0]
                    drawing.arucoDetectedMarkers(frame, [corners[idx]], ids[idx].reshape((1,1)), sub_pixel_fac=self.sub_pixel_fac, special_highlight=[[m_id],(255,0,255)])
                if self.show_individual_marker_axes and self.cam_params.has_intrinsics():
                    individual_marker_out[m_id].draw_frame_axis(frame, self.cam_params, self.individual_markers[m_id]['marker_size']/2, self.sub_pixel_fac)

            # draw output of extra processing functions
            for e in extra_processing_for_this_frame:
                if self.show_sync_func_output and self.extra_proc_visualizer[e]:
                    self.extra_proc_visualizer[e](frame, *extra_processing_out[e])

        if self.has_gui:
            self.gui.update_image(frame, frame_ts/1000., frame_idx)

        self._cache = Status.Ok, pose_out, individual_marker_out, extra_processing_out, (frame, frame_idx, frame_ts)
        return self._cache

    def process_video(self) -> tuple[dict[str, list[plane.Pose]], dict[str, list[marker.Pose]], dict[str, list[list[int, Any]]]]:
        poses_out               : dict[str, list[plane.Pose]]       = {p:[] for p in self.planes}
        individual_markers_out  : dict[str, list[marker.Pose]]      = {i:[] for i in self.individual_markers}
        extra_processing_out    : dict[str, list[list[int, Any]]]   = {e:[] for e in self.extra_proc_functions}
        while True:
            status, pose, individual_marker, extra_proc, _ = self.process_one_frame()
            if status==Status.Finished:
                break
            if status==Status.Skip:
                continue
            # store outputs
            for p in pose:
                poses_out[p].append(pose[p])
            for i in individual_marker:
                individual_markers_out[i].append(individual_marker[i])
            for e in extra_proc:
                extra_processing_out[e].append(extra_proc[e])

        return poses_out, individual_markers_out, extra_processing_out