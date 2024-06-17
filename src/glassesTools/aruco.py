import numpy as np
import cv2
from typing import Any

from . import drawing, intervals, marker, ocv, plane, timestamps



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

    def estimate_pose(self, corners, ids) -> tuple[int, np.ndarray, np.ndarray]:
        objP, imgP = self._match_image_points(corners, ids)
        return self._estimate_pose_impl(objP, imgP)

    def _estimate_pose_impl(self, objP, imgP):
        # NB: N_markers also flags success of the pose estimation. Also 0 if not successful or not possible (missing intrinsics)
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
        # NB: N_markers also flags success of the pose estimation. Also 0 if not successful or not possible (missing intrinsics)
        from . import transforms
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
    def estimate_pose_and_homography(self, frame_idx, min_num_markers, corners, ids) -> tuple[plane.Pose, dict[str,Any]]:
        pose = plane.Pose(frame_idx)
        if ids is not None and len(ids) >= min_num_markers:
            # get matching image and board points
            objP, imgP = self._match_image_points(corners, ids)

            # get camera pose
            pose.pose_N_markers, pose.pose_R_vec, pose.pose_T_vec = \
                self._estimate_pose_impl(objP, imgP)

            # also get homography (direct image plane to plane in world transform)
            pose.homography_N_markers, pose.homography_mat = \
                self._estimate_homography_impl(objP, imgP)
        return pose

    def detect_and_estimate(self, frame, frame_idx, min_num_markers) -> tuple[plane.Pose, dict[str,Any]]:
        corners, ids, rejectedImgPoints, recoveredIds = self.detect_markers(frame, min_nmarker_refine=min_num_markers)
        pose = self.estimate_pose_and_homography(frame_idx, min_num_markers, corners, ids)
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

def run_pose_estimation(in_video, frame_timestamp_file, camera_calibration_file,
                        output_dir,
                        processing_intervals,
                        planes, individual_markers,
                        gui, sub_pixel_fac = 8, show_rejected_markers = False) -> tuple[bool, dict[str,list[plane.Pose]], dict[str,list[marker.Pose]]]:
    show_visualization = gui is not None

    # open video
    cap = ocv.CV2VideoReader(in_video, timestamps.VideoTimestamps(frame_timestamp_file).timestamps)

    # setup aruco marker detection
    aruco_boards = {p: planes[p]['plane'].get_aruco_board() for p in planes}
    detectors = {p: ArUcoDetector(aruco_boards[p].getDictionary(), planes[p]['aruco_params']) for p in planes}
    cam_params = ocv.CameraParams.readFromFile(camera_calibration_file)
    for p in detectors:
        detectors[p].set_board(aruco_boards[p])
        # get camera calibration info
        detectors[p].set_intrinsics(cam_params)

    # check if we can do an optimization of detecting the markers only once for multiple planes (if it makes sense because we have more than one plane)
    plane_names = list(planes.keys())
    aruco_dicts = [planes[p]['aruco_dict'] for p in planes if 'aruco_dict' in planes[p]]
    all_same_dict_and_dect = len(planes)>1 and len(aruco_dicts)==len(planes) and len(set(aruco_dicts))==1 and all([planes[plane_names[0]]['aruco_params']==planes[p]['aruco_params'] for p in planes])

    # check individual markers (detected using the same detector as for the plane(s))
    has_individual_markers = individual_markers is not None
    if has_individual_markers:
        assert all_same_dict_and_dect or len(planes)==1, "Detecting and Reporting individual markers are only supported when there is a single plane, or all planes have identical aruco setup"
        individual_markers_out = {i:[] for i in individual_markers}
        object_points = {}
        for i in individual_markers:
            marker_size = individual_markers[i]['marker_size']
            object_points[i] = np.array([[-marker_size/2, marker_size/2, 0],[marker_size/2, marker_size/2, 0],[marker_size/2, -marker_size/2, 0],[-marker_size/2, -marker_size/2, 0]])
    else:
        individual_markers_out = None

    stop_all_processing = False
    poses = {p:[] for p in planes}
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
        planes_for_this_frame = [p for p in planes if intervals.is_in_interval(frame_idx, processing_intervals[p])]
        if not planes_for_this_frame:
            continue

        # detect markers
        detect_dicts = {}
        if all_same_dict_and_dect or has_individual_markers:
            corners, ids, rejected_corners = _detect_markers(frame, detectors[plane_names[0]]._det)
            detect_dicts = _refine_for_multiple_planes(frame, corners, ids, rejected_corners, {p:detectors[p] for p in planes_for_this_frame}, cam_params.camera_mtx, cam_params.distort_coeffs)
        else:
            for p in planes_for_this_frame:
                detect_dicts[p] = dict(zip(['corners', 'ids', 'rejectedImgPoints', 'recoveredIds'], detectors[p].detect_markers(frame, planes[p]['min_num_markers'])))
        # determine pose
        for p in planes_for_this_frame:
            pose = detectors[p].estimate_pose_and_homography(frame_idx, planes[p]['min_num_markers'], detect_dicts[p]['corners'], detect_dicts[p]['ids'])
            poses[p].append(pose)
            # draw detection and pose, if wanted
            if show_visualization:
                detectors[p].visualize(frame, pose, detect_dicts[p], planes[p]['plane'].marker_size/2, sub_pixel_fac, show_rejected_markers)

        # deal with individual markers, if any
        if has_individual_markers and ids is not None:
            found_markers = np.where([x[0] in individual_markers for x in ids])[0]
            if found_markers.size>0:
                for idx in found_markers:
                    m_id = ids[idx][0]
                    pose = marker.Pose(frame_idx)
                    if cam_params.has_intrinsics():
                        # can only get marker pose if we have a calibrated camera (need intrinsics), else at least flag that marker was found
                        _, pose.R_vec, pose.T_vec = cv2.solvePnP(object_points[m_id], corners[idx], cam_params.camera_mtx, cam_params.distort_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    individual_markers_out[m_id].append(pose)
                    if show_visualization and cam_params.has_intrinsics():
                        pose.draw_origin_on_frame(frame, cam_params.camera_mtx, cam_params.distort_coeffs, individual_markers[m_id]['marker_size']/2, sub_pixel_fac)

        if show_visualization:
            # keys is populated above
            if 's' in keys:
                # screenshot
                cv2.imwrite(output_dir / f'detect_frame_{frame_idx}.png', frame)
            gui.update_image(frame, frame_ts/1000., frame_idx)
            closed, = gui.get_state()
            if closed:
                stop_all_processing = True
                break

    if show_visualization:
        gui.stop()

    return stop_all_processing, poses, individual_markers_out