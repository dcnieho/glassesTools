import numpy as np
import cv2
from typing import Any

from . import drawing, intervals, ocv, plane, timestamps, transforms


class ArUcoDetector():
    def __init__(self, ArUco_dict: cv2.aruco.Dictionary, params: dict[str, Any]):
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
        corners, ids, rejected_corners = self._det.detectMarkers(image)
        recovered_ids = None
        if np.any(ids==None):
            ids = None

        # Refine detected markers (eliminates markers not part of our poster, adds missing markers to the poster)
        if self._board and ids is not None and min_nmarker_refine and len(ids)>min_nmarker_refine:
            corners, ids, rejected_corners, recovered_ids = \
                self.refine_detected_markers(image, corners, ids, rejected_corners)
        return corners, ids, rejected_corners, recovered_ids

    def refine_detected_markers(self, image, detected_corners, detected_ids, rejected_corners):
        corners, ids, rejectedImgPoints, recoveredIds = self._det.refineDetectedMarkers(
                                image = image, board = self._board,
                                detectedCorners = detected_corners, detectedIds = detected_ids, rejectedCorners = rejected_corners,
                                cameraMatrix = self._camera_params.camera_mtx, distCoeffs = self._camera_params.distort_coeffs)
        if corners and corners[0].shape[0]==4:
            # there are versions out there where there is a bug in output shape of each set of corners, fix up
            corners = [np.reshape(c,(1,4,2)) for c in corners]
        if rejectedImgPoints and rejectedImgPoints[0].shape[0]==4:
            # same as for corners
            rejectedImgPoints = [np.reshape(c,(1,4,2)) for c in rejectedImgPoints]

        return corners, ids, rejectedImgPoints, recoveredIds

    def _match_image_points(self, corners, ids):
        return self._board.matchImagePoints(corners, ids) # -> objP, imgP

    def estimate_pose(self, corners, ids) -> tuple[int, np.ndarray, np.ndarray]:
        objP, imgP = self._match_image_points(corners, ids)
        return self._estimate_pose_impl(objP, imgP)

    def _estimate_pose_impl(self, objP, imgP):
        N_markers, R_vec, T_vec = 0, None, None
        if objP is None or not self._camera_params.has_intrinsics():
            return N_markers, R_vec, T_vec

        ok, R_vec, T_vec = cv2.solvePnP(objP, imgP, self._camera_params.camera_mtx, self._camera_params.distort_coeffs, np.empty(1), np.empty(1))
        if ok:
            N_markers = int(objP.shape[0]/4)
        return N_markers, R_vec, T_vec

    def estimate_homography(self, corners, ids) -> tuple[np.ndarray, bool]:
        objP, imgP = self._match_image_points(corners, ids)
        return self._estimate_homography_impl(objP, imgP)

    def _estimate_homography_impl(self, objP, imgP):
        N_markers, H = 0, None
        if objP is None:
            return N_markers, H

        # use undistorted marker corners if possible
        if self._camera_params.has_intrinsics():
            imgP = np.vstack([cv2.undistortPoints(x, self._camera_params.camera_mtx, self._camera_params.distort_coeffs, P=self._camera_params.camera_mtx) for x in imgP])

        H, status = transforms.estimateHomography(objP, imgP)
        if status:
            N_markers = int(objP.shape[0]/4)
        return N_markers, H

    # higher level functions for detecting + pose estimation, and for results visualization
    def detect_and_estimate(self, frame, frame_idx, min_num_markers) -> tuple[plane.Pose, dict[str,Any]]:
        pose = plane.Pose(frame_idx)
        corners, ids, rejectedImgPoints, recoveredIds = self.detect_markers(frame, min_nmarker_refine=min_num_markers)

        if ids is not None and len(ids) >= min_num_markers:
            # get matching image and board points
            objP, imgP = self._match_image_points(corners, ids)

            # get camera pose
            pose.pose_N_markers, pose.pose_R_vec, pose.pose_T_vec = \
                self._estimate_pose_impl(objP, imgP)

            # also get homography (direct image plane to plane in world transform)
            pose.homography_N_markers, pose.homography_mat = \
                self._estimate_homography_impl(objP, imgP)

        return pose, {'corners': corners, 'ids': ids, 'rejectedImgPoints': rejectedImgPoints, 'recoveredIds': recoveredIds}

    def visualize(self, frame, pose: plane.Pose, detect_dict, arm_length, sub_pixel_fac = 8, show_rejected_markers = False):
        if pose.pose_N_markers>0:
            # draw axis indicating poster pose (origin and orientation)
            drawing.openCVFrameAxis(frame, self._camera_params.camera_mtx, self._camera_params.distort_coeffs, pose.pose_R_vec, pose.pose_T_vec, arm_length, 3, sub_pixel_fac)

        if pose.homography_N_markers>0:
            # find where plane origin is expected to be in the image
            target = pose.planeToCamHomography([0., 0.], self._camera_params)
            # draw target location on image
            if target[0] >= 0 and target[0] < frame.shape[1] and target[1] >= 0 and target[1] < frame.shape[0]:
                drawing.openCVCircle(frame, target, 3, (0,0,0), -1, sub_pixel_fac)

        # if any markers were detected, draw where on the frame
        drawing.arucoDetectedMarkers(frame, detect_dict['corners'], detect_dict['ids'], subPixelFac=sub_pixel_fac, specialHighlight=[detect_dict['recoveredIds'],(255,255,0)])

        # for debug, can draw rejected markers on frame
        if show_rejected_markers:
            cv2.aruco.drawDetectedMarkers(frame, detect_dict['rejectedImgPoints'], None, borderColor=(211,0,148))



def create_board(board_corner_points: list[np.ndarray], ids: list[int], ArUco_dict: cv2.aruco.Dictionary):
    board_corner_points = np.dstack(board_corner_points)        # list of 2D arrays -> 3D array
    board_corner_points = np.rollaxis(board_corner_points,-1)   # 4x2xN -> Nx4x2
    board_corner_points = np.pad(board_corner_points,((0,0),(0,0),(0,1)),'constant', constant_values=(0.,0.)) # Nx4x2 -> Nx4x3
    return cv2.aruco.Board(board_corner_points, ArUco_dict, np.array(ids))

def run_pose_estimation(in_video, frame_timestamp_file, camera_calibration_file,
                        output_dir, out_file,
                        processing_intervals,
                        ArUco_board, aruco_params, min_num_markers,
                        gui, arm_length, sub_pixel_fac = 8, show_rejected_markers = False):
    show_visualization = gui is not None

    # open video
    cap = ocv.CV2VideoReader(in_video, timestamps.from_file(frame_timestamp_file))

    # setup aruco marker detection
    detector = ArUcoDetector(ArUco_board.getDictionary(), aruco_params)
    detector.set_board(ArUco_board)
    # get camera calibration info
    detector.set_intrinsics(ocv.CameraParams.readFromFile(camera_calibration_file))


    stop_all_processing = False
    poses = []
    while True:
        # process frame-by-frame
        done, frame, frame_idx, frame_ts = cap.read_frame(report_gap=True)
        if done or intervals.beyond_last_interval(frame_idx, processing_intervals):
            break
        cap.report_frame()

        if frame is None:
            # we don't have a valid frame, continue to next
            continue

        if show_visualization:
            keys = gui.get_key_presses()
            if 'q' in keys:
                # quit fully
                stop_all_processing = True
                break
            if 'n' in keys:
                # goto next
                break

        # check we're in a current interval, else skip processing
        # NB: have to spool through like this, setting specific frame to read
        # with cap.set(cv2.CAP_PROP_POS_FRAMES) doesn't seem to work reliably
        # for VFR video files
        if not intervals.is_in_interval(frame_idx, processing_intervals):
            continue

        # detect markers
        pose, detect_dict = detector.detect_and_estimate(frame, frame_idx, min_num_markers=min_num_markers)
        poses.append(pose)
        # draw detection and pose, if wanted
        if show_visualization:
            detector.visualize(frame, pose, detect_dict, arm_length, sub_pixel_fac, show_rejected_markers)

        if show_visualization:
            # keys is populated above
            if 's' in keys:
                # screenshot
                cv2.imwrite(str(output_dir / ('detect_frame_%d.png' % frame_idx)), frame)
            gui.update_image(frame, frame_ts/1000., frame_idx)
            closed, = gui.get_state()
            if closed:
                stop_all_processing = True
                break

    plane.Pose.writeToFile(poses, output_dir / out_file, skip_failed=True)

    if show_visualization:
        gui.stop()

    return stop_all_processing