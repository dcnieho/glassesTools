import numpy as np
import cv2
import pathlib
from typing import Any, TypedDict

from . import drawing, marker, ocv, plane, pose

default_dict = cv2.aruco.DICT_4X4_250

dict_id_to_str: dict[int,str] = {getattr(cv2.aruco,k):k for k in ['DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000', 'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000', 'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000', 'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000', 'DICT_ARUCO_ORIGINAL', 'DICT_APRILTAG_16H5', 'DICT_APRILTAG_25H9', 'DICT_APRILTAG_36H10', 'DICT_APRILTAG_36H11', 'DICT_ARUCO_MIP_36H12']}
def str_to_dict_id(aruco_dict_name: str):
    if not hasattr(cv2.aruco,aruco_dict_name):
        raise ValueError(f'ArUco dictionary with name "{aruco_dict_name}" is not known.')
    return getattr(cv2.aruco,aruco_dict_name)

# same number means that the dictionaries are the same (i.e. marker 3 is the same for all the dictionaries of the same family), just different number of markers in the dictionary
dict_id_to_family = {
    cv2.aruco.DICT_4X4_50:          0,
    cv2.aruco.DICT_4X4_100:         0,
    cv2.aruco.DICT_4X4_250:         0,
    cv2.aruco.DICT_4X4_1000:        0,
    cv2.aruco.DICT_5X5_50:          1,
    cv2.aruco.DICT_5X5_100:         1,
    cv2.aruco.DICT_5X5_250:         1,
    cv2.aruco.DICT_5X5_1000:        1,
    cv2.aruco.DICT_6X6_50:          2,
    cv2.aruco.DICT_6X6_100:         2,
    cv2.aruco.DICT_6X6_250:         2,
    cv2.aruco.DICT_6X6_1000:        2,
    cv2.aruco.DICT_7X7_50:          3,
    cv2.aruco.DICT_7X7_100:         3,
    cv2.aruco.DICT_7X7_250:         3,
    cv2.aruco.DICT_7X7_1000:        3,
    cv2.aruco.DICT_ARUCO_ORIGINAL:  4,
    cv2.aruco.DICT_APRILTAG_16H5:   5,
    cv2.aruco.DICT_APRILTAG_25H9:   6,
    cv2.aruco.DICT_APRILTAG_36H10:  7,
    cv2.aruco.DICT_APRILTAG_36H11:  8,
    cv2.aruco.DICT_ARUCO_MIP_36H12: 9
}
family_to_str = {
    0: ('DICT_4X4', True),
    1: ('DICT_5X5', True),
    2: ('DICT_6X6', True),
    3: ('DICT_7X7', True),
    4: ('DICT_ARUCO_ORIGINAL', False),
    5: ('DICT_APRILTAG_16H5', False),
    6: ('DICT_APRILTAG_25H9', False),
    7: ('DICT_APRILTAG_36H10', False),
    8: ('DICT_APRILTAG_36H11', False),
    9: ('DICT_ARUCO_MIP_36H12', False),
}

class PlaneSetup(TypedDict):
    plane                   : plane.Plane
    aruco_detector_params   : dict[str,Any]
    aruco_refine_params     : dict[str,Any]
    min_num_markers         : int
class MarkerSetup(TypedDict):
    aruco_detector_params   : dict[str,Any]
    detect_only             : bool
    size                    : float

def reduce_to_families(dictionary_ids: list[int]) -> tuple[list[int],dict[int,int]]:
    # turn into families, and the largest dict necessary per family
    # get unique dictionaries
    seen: set[int] = set()
    aruco_dicts = [x for x in dictionary_ids if x not in seen and not seen.add(x)]
    # first organize by family
    by_family: dict[int,list[int]] = {}
    for d in aruco_dicts:
        f = dict_id_to_family[d]
        if f not in by_family:
            by_family[f] = []
        by_family[f].append(d)
    # for each family, if there are multiple dicts, get the largest
    needed_dicts = [sorted(by_family[f], key=lambda x: get_dict_size(x))[-1] for f in by_family]
    # make a mapping of dictionary (requested) to dictionary (used)
    aruco_dict_mapping = {d:d2 for f,d2 in zip(by_family,needed_dicts) for d in by_family[f]}
    return needed_dicts, aruco_dict_mapping

def get_dict_size(dictionary_id: int) -> int:
    return cv2.aruco.getPredefinedDictionary(dictionary_id).bytesList.shape[0]

def get_marker_image(size: int, m_id: int, ArUco_dict_id: int, marker_border_bits: int) -> np.ndarray|None:
    if m_id>=get_dict_size(ArUco_dict_id):
        return None
    marker_image = np.zeros((size, size), dtype=np.uint8)
    return cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ArUco_dict_id), m_id, size, marker_image, marker_border_bits)

def deploy_marker_images(output_dir: str|pathlib.Path, size: int, ArUco_dict_id: int, marker_border_bits: int=1):
    # Generate the markers
    for m_id in range(get_dict_size(ArUco_dict_id)):
        marker_image = get_marker_image(size, m_id, ArUco_dict_id, marker_border_bits)
        if marker_image is not None:
            cv2.imwrite(output_dir / f"{m_id}.png", marker_image)

class Detector:
    def __init__(self, dictionary_id: int):
        self.dictionary_id  = dictionary_id
        self._family        = dict_id_to_family[self.dictionary_id]
        self._is_family     = family_to_str[self._family][1]

        self.planes             : dict[str, PlaneSetup]         = {}
        self._boards            : dict[str, cv2.aruco.Board]    = {}
        self.individual_markers : dict[int, MarkerSetup]        = {}
        self._indiv_marker_points:dict[int, np.ndarray]         = {}

        self._plane_marker_ids      : dict[str,set[int]]        = {}
        self._individual_marker_ids : set[int]                  = set()
        self._all_markers           : set[int]                  = set()

        self._user_detector_params  : dict[str]                 = {}
        self._user_refine_params    : dict[str]                 = {}

        self._det: cv2.aruco.ArucoDetector                      = None

        self._last_detect_output : tuple[dict[str,dict[str]],dict[str],dict[str],list[np.ndarray]] = {}

    def add_plane(self, name: str, setup: PlaneSetup):
        self._check_dict(setup['plane'].aruco_dict_id, 'plane')
        self.planes[name] = setup
        if 'aruco_detector_params' in self.planes[name] and self.planes[name]['aruco_detector_params']:
            self._update_parameters('detector', self.planes[name]['aruco_detector_params'])
        if 'aruco_refine_params' in self.planes[name] and self.planes[name]['aruco_refine_params']:
            self._update_parameters('refine'  , self.planes[name]['aruco_refine_params'])
        self._boards[name]= self.planes[name]['plane'].get_aruco_board()

        markers = self.planes[name]['plane'].get_marker_IDs()
        for ms in markers:
            if ms!='plane':
                continue
            m_ids = {m.m_id for m in markers[ms]}
            self._all_markers.update(m_ids)
            self._plane_marker_ids[name] = m_ids

    def add_individual_marker(self, mark: marker.MarkerID, setup: MarkerSetup):
        self._check_dict(mark.aruco_dict_id, 'individual marker')
        self.individual_markers[mark.m_id] = setup
        if 'aruco_detector_params' in self.individual_markers[mark.m_id] and self.individual_markers[mark.m_id]['aruco_detector_params']:
            self._update_parameters('detector', self.individual_markers[mark.m_id]['aruco_detector_params'])
        self._all_markers.add(mark.m_id)
        self._individual_marker_ids.add(mark.m_id)
        # get marker points in world
        marker_size = self.individual_markers[mark.m_id].get('size',None) if not self.individual_markers[mark.m_id].get('detect_only',False) else None
        if not marker_size or marker_size<0.:
            marker_points = None
        else:
            marker_points =  np.array([[-marker_size/2,  marker_size/2, 0],
                                       [ marker_size/2,  marker_size/2, 0],
                                       [ marker_size/2, -marker_size/2, 0],
                                       [-marker_size/2, -marker_size/2, 0]])
        self._indiv_marker_points[mark.m_id] = marker_points

    def _check_dict(self, dict_id: int, what: str):
        if self._is_family:
            family = dict_id_to_family[dict_id]
            if family!=self._family:
                raise ValueError(f'The dictionary for this new {what}, {dict_id_to_str[dict_id]}, is not part of the family ({family_to_str[family][0]}) used for this detector. Use dictionary {dict_id_to_str[self.dictionary_id]} or smaller.')
            elif dict_id>self.dictionary_id:
                raise ValueError(f'The dictionary for this new {what}, {dict_id_to_str[dict_id]}, contains more markers than the dictionary used for this detector ({dict_id_to_str[self.dictionary_id]}). Use a dictionary with more markers when creating this detector.')
        elif dict_id!=self.dictionary_id:
            raise ValueError(f'The dictionary for this new {what}, {dict_id_to_str[dict_id]}, does not match the dictionary used for this detector ({dict_id_to_str[self.dictionary_id]}).')

    def _update_parameters(self, which: str, new_params: dict):
        if which=='detector':
            param_dict = self._user_detector_params
            cls = cv2.aruco.DetectorParameters
        elif 'refine':
            param_dict = self._user_refine_params
            cls = cv2.aruco.RefineParameters
        else:
            raise ValueError(f'parameter type "{which}" not understood')
        for p in new_params:
            if not hasattr(cls, p):
                raise AttributeError(f'{p} is not a valid parameter for cv2.aruco.{cls.__name__}')
            if p in param_dict and new_params[p]!=param_dict[p]:
                fam_str,is_family = family_to_str[dict_id_to_family[self.dictionary_id]]
                dict_str = f'{fam_str} family' if is_family else f'{dict_id_to_str[self.dictionary_id]} dictionary'
                raise ValueError(f'You have already set the parameter {p} to {param_dict[p]} and are now trying to set it to {new_params[p]}, in the detector for the {dict_str}. Resolve this conflict by checking this setting for all planes and individual markers using the {dict_str}.')
            param_dict[p] = new_params[p]

    def create_detector(self):
        # set detector parameters
        detector_params                       = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod= cv2.aruco.CORNER_REFINE_SUBPIX    # good default, user can override
        refine_params                         = cv2.aruco.RefineParameters()
        for p in self._user_detector_params:
            setattr(detector_params, p, self._user_detector_params[p])
        for p in self._user_refine_params:
            setattr(refine_params, p, self._user_refine_params[p])

        self._det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(self.dictionary_id),
                                            detector_params, refine_params)

    def detect_markers(self, image: cv2.UMat, camera_params: ocv.CameraParams) -> tuple[dict[str,dict[str]],dict[str],dict[str],list[np.ndarray]]:
        img_points, ids, rejected_img_points = self._detect_markers(image, self._det)

        # For each plane, refine detected markers (eliminates markers not part of the plane, adds missing markers to the poster)
        out_planes: dict[str] = {}
        for p in self.planes:
            if ids is not None and len(ids)>self.planes[p]['min_num_markers']:
                img_points, ids, rejected_img_points, recovered_ids = \
                    self._refine_detection(image, img_points, ids, rejected_img_points,
                                           self._det, self._boards[p],
                                           camera_params.camera_mtx, camera_params.distort_coeffs)
                out_planes[p] = dict(zip(['img_points', 'ids', 'recovered_ids'],(img_points, ids, recovered_ids)))
            else:
                out_planes[p] = None if ids is None else dict(zip(['img_points', 'ids', 'recovered_ids'],(img_points, ids, None)))
            if out_planes[p]:
                out_planes[p]['img_points'],out_planes[p]['ids'] = self._filter_detections(out_planes[p]['img_points'], out_planes[p]['ids'], self._plane_marker_ids[p])

        # For individual markers, only keep the ones known
        out_individual: dict[str] = {}
        out_individual['img_points'],out_individual['ids'] = self._filter_detections(img_points, ids, self._individual_marker_ids)

        # collect unexpected markers
        unexpected_markers: dict[str] = {}
        unexpected_markers['img_points'],unexpected_markers['ids'] = self._filter_detections(img_points, ids, self._all_markers, keep_expected=False)

        self._last_detect_output = (out_planes, out_individual, unexpected_markers, rejected_img_points)
        return self._last_detect_output

    def _detect_markers(self, image: cv2.UMat, det: cv2.aruco.ArucoDetector):
        img_points, ids, rejected_img_points = det.detectMarkers(image)
        if np.any(ids==None):
            ids = None
        return img_points, ids, rejected_img_points

    def _refine_detection(self, image: cv2.UMat, detected_corners, detected_ids, rejected_corners, det: cv2.aruco.ArucoDetector, board: cv2.aruco.Board, camera_mtx, distort_coeffs):
        img_points, ids, rejected_img_points, recovered_ids = \
            det.refineDetectedMarkers(
                image = image, board = board,
                detectedCorners = detected_corners, detectedIds = detected_ids, rejectedCorners = rejected_corners,
                cameraMatrix = camera_mtx, distCoeffs = distort_coeffs
                )
        if img_points and img_points[0].shape[0]==4:
            # there are versions out there where there is a bug in output shape of each set of corners, fix up
            img_points = [np.reshape(c,(1,4,2)) for c in img_points]
        if rejected_img_points and rejected_img_points[0].shape[0]==4:
            # same as for corners
            rejected_img_points = [np.reshape(c,(1,4,2)) for c in rejected_img_points]

        return img_points, ids, rejected_img_points, recovered_ids

    def _filter_detections(self, img_points: list[np.ndarray], ids: np.ndarray, expected_ids: list[np.ndarray], keep_expected=True):
        if ids is None or not img_points:
            return img_points, ids
        if not keep_expected:
            expected_ids = set(ids.flatten())-set(expected_ids)
        if not expected_ids:
            # optimization. If output will definitely be empty, return directly
            return tuple(), None
        to_remove = np.where([x not in expected_ids for x in ids.flatten()])[0]
        ids = np.delete(ids, to_remove, axis=0)
        img_points = tuple(v for i,v in enumerate(img_points) if i not in to_remove)
        return img_points, ids

    def get_matching_image_board_points(self, plane_name: str, detect_tuple=None):
        if detect_tuple is None:
            detect_tuple = self._last_detect_output
        if plane_name not in detect_tuple[0] or detect_tuple[0][plane_name]['ids'] is None or not detect_tuple[0][plane_name]['img_points']:
            return None, None
        objP, imgP = self._boards[plane_name].matchImagePoints(detect_tuple[0][plane_name]['img_points'], detect_tuple[0][plane_name]['ids'])
        if imgP is None or int(imgP.shape[0]/4)<self.planes[plane_name]['min_num_markers']:
            return None, None
        return imgP, objP

    def get_individual_marker_points(self, marker_id: int, detect_tuple=None):
        if detect_tuple is None:
            detect_tuple = self._last_detect_output
        if detect_tuple[1]['ids'] is None or not detect_tuple[1]['img_points'] or marker_id not in detect_tuple[1]['ids']:
            return None, None
        img_points = detect_tuple[1]['img_points'][detect_tuple[1]['ids'].flatten().tolist().index(marker_id)]
        return img_points, self._indiv_marker_points[marker_id]

    def visualize(self, frame, detect_tuple=None, sub_pixel_fac=8, plane_marker_color=(0,255,0), recovered_plane_marker_color=(255,255,0), individual_marker_color=(255,0,255), unexpected_marker_color=(150,253,253), rejected_marker_color=None):
        if detect_tuple is None:
            detect_tuple = self._last_detect_output
        special_highlight = []

        # for debug, can draw rejected markers on frame
        if rejected_marker_color is not None:
            cv2.aruco.drawDetectedMarkers(frame, detect_tuple[3], None, borderColor=rejected_marker_color)

        # draw detected markers on the frame
        if plane_marker_color is not None:
            for p in detect_tuple[0]:
                if not detect_tuple[0][p] or 'ids' not in detect_tuple[0][p] or len(detect_tuple[0][p]['ids'])==0:
                    continue
                if recovered_plane_marker_color is not None and detect_tuple[0][p]['recovered_ids'] is not None and len(detect_tuple[0][p]['recovered_ids'])>0:
                    special_highlight = [detect_tuple[0][p]['recovered_ids'],recovered_plane_marker_color]
                drawing.arucoDetectedMarkers(frame, detect_tuple[0][p]['img_points'], detect_tuple[0][p]['ids'], border_color=plane_marker_color, sub_pixel_fac=sub_pixel_fac, special_highlight=special_highlight)
        if individual_marker_color is not None and detect_tuple[1]['ids'] is not None and len(detect_tuple[1]['ids']>0):
            drawing.arucoDetectedMarkers(frame, detect_tuple[1]['img_points'], detect_tuple[1]['ids'], border_color=individual_marker_color, sub_pixel_fac=sub_pixel_fac)
        if unexpected_marker_color is not None and detect_tuple[2]['ids'] is not None and len(detect_tuple[2]['ids']>0):
            drawing.arucoDetectedMarkers(frame, detect_tuple[2]['img_points'], detect_tuple[2]['ids'], border_color=unexpected_marker_color, sub_pixel_fac=sub_pixel_fac)

class Manager:
    # takes single planes and individual markers, and for each information about when they should be
    # detected, and consolidates them into a minimal set of detectors
    # with all planes/individual markers associated to one of these detectors
    def __init__(self):
        # planes to be detected
        self.planes                 : dict[str, PlaneSetup]                 = {}
        self.plane_proc_intervals   : dict[str, list[int]|list[list[int]]]  = {}
        self._plane_to_detector     : dict[str, int]                        = {}
        # individual markers to be detected
        self.individual_markers                 : dict[marker.MarkerID, MarkerSetup]    = {}
        self.individual_markers_proc_intervals  : dict[str, list[int]|list[list[int]]]  = {}

        # consolidated into set of detectors, and associated planes+individual markers for each
        self._detectors             : dict[int, Detector]                   = {}
        self._det_cache             : dict[int, tuple[int,tuple]]           = {}
        self._last_viz_frame_idx    : dict[int,int]                         = {}

        # colors for drawing (in BGR order)
        self._plane_marker_color            = (  0,255,  0)
        self._recovered_plane_marker_color  = (255,255,  0)
        self._individual_marker_color       = (255,  0,255)
        self._unexpected_marker_color       = (128,255,255)
        self._rejected_marker_color         = None # by default not drawn. If wanted, (0,0,255) is a good color

    def add_plane(self, plane: str, planes_setup: PlaneSetup, processing_intervals: list[int]|list[list[int]] = None):
        if plane in self.planes:
            raise ValueError(f'Cannot register the plane "{plane}", it is already registered')
        self.planes[plane]                  = planes_setup
        self.plane_proc_intervals[plane]    = processing_intervals

    def add_individual_marker(self, mark: marker.MarkerID, marker_setup: MarkerSetup, processing_intervals: list[int]|list[list[int]] = None):
        if mark in self.individual_markers:
            raise ValueError(f'Cannot register the individual marker {marker.marker_ID_to_str(mark)}, it is already registered')
        self.individual_markers[mark]                = marker_setup
        self.individual_markers_proc_intervals[mark] = processing_intervals

    def set_visualization_colors(self, plane_marker_color=(0,255,0), recovered_plane_marker_color=(0,255,255), individual_marker_color=(255,0,255), unexpected_marker_color=(255,255,128), rejected_marker_color=None):
        # user should provide colors in RGB, internally we store as BGR
        if plane_marker_color is not None:
            plane_marker_color = plane_marker_color[::-1]
        self._plane_marker_color = plane_marker_color
        if recovered_plane_marker_color is not None:
            recovered_plane_marker_color = recovered_plane_marker_color[::-1]
        self._recovered_plane_marker_color = recovered_plane_marker_color
        if individual_marker_color is not None:
            individual_marker_color = individual_marker_color[::-1]
        self._individual_marker_color = individual_marker_color
        if unexpected_marker_color is not None:
            unexpected_marker_color = unexpected_marker_color[::-1]
        self._unexpected_marker_color = unexpected_marker_color
        if rejected_marker_color is not None:
            rejected_marker_color = rejected_marker_color[::-1]
        self._rejected_marker_color = rejected_marker_color

    def consolidate_setup(self):
        # get list of all ArUco dicts and markers we're dealing with
        all_markers: set[marker.MarkerID] = set()
        for p in self.planes:
            markers = self.planes[p]['plane'].get_marker_IDs()
            for ms in markers:
                if ms!='plane':
                    # N.B.: other markers should be registered by caller as individual markers
                    continue
                if all_markers.intersection(markers[ms]):
                    raise RuntimeError('Markers are not unique')
                all_markers.update(markers[ms])
        for m in self.individual_markers:
            if m in all_markers:
                raise RuntimeError('Markers are not unique')
            all_markers.add(m)

        # see for which marker dicts we need detectors to service all these
        # also determine mapping of requested ArUco dicts to these detectors
        needed_dicts, dict_mapping = reduce_to_families({m.aruco_dict_id for m in all_markers})

        # organize planes and individual markers into the dict that will be used for their detection
        planes_organized        : dict[int,list[str]]             = {d:[] for d in needed_dicts}
        indiv_markers_organized : dict[int,list[marker.MarkerID]] = {d:[] for d in needed_dicts}
        for p in self.planes:
            det_dict = dict_mapping[self.planes[p]['plane'].aruco_dict_id]
            planes_organized[det_dict].append(p)
        for m in self.individual_markers:
            det_dict = dict_mapping[m.aruco_dict_id]
            indiv_markers_organized[det_dict].append(m)

        # make the needed detectors
        self._detectors.clear()
        self._plane_to_detector.clear()
        for d in needed_dicts:
            self._detectors[d] = Detector(d)
            for p in planes_organized[d]:
                self._detectors[d].add_plane(p, self.planes[p])
                self._plane_to_detector[p] = d
            for m in indiv_markers_organized[d]:
                self._detectors[d].add_individual_marker(m, self.individual_markers[m])
            self._detectors[d].create_detector()

    def register_with_estimator(self, estimator: pose.Estimator):
        # this handles registration of all planes and individual markers with the estimator
        # and makes sure our wrapper function for the aruco detector gets called which handles
        # aruco detection so that each detector is run only once on a frame
        for p in self.planes:
            estimator.add_plane(p,
                                lambda pn, fi, fr, cp: self._detect_plane(pn, fi, fr, cp),
                                self.plane_proc_intervals[p],
                                lambda pn, fi, fr, _: self._visualize_plane(pn, fi, fr))
        for m in self.individual_markers:
            estimator.add_individual_marker(m,
                                            lambda k, fi, fr, cp: self._detect_individual_marker(k, fi, fr, cp),
                                            self.individual_markers_proc_intervals[m],
                                            lambda k, fi, fr, _: self._visualize_individual_marker(k, fi, fr))

    def _detect_plane(self, plane_name: str, frame_idx: int, frame: np.ndarray, camera_parameters: ocv.CameraParams):
        if plane_name not in self._plane_to_detector:
            raise ValueError(f'The plane {plane_name} is not known')
        aruco_dict_id = self._plane_to_detector[plane_name]
        detect_tuple = self._get_detector_cache(aruco_dict_id, frame_idx, frame, camera_parameters)
        if not detect_tuple[0] or plane_name not in detect_tuple[0] or not detect_tuple[0][plane_name]:
            return None, None
        return self._detectors[aruco_dict_id].get_matching_image_board_points(plane_name, detect_tuple)

    def _detect_individual_marker(self, mark: marker.MarkerID, frame_idx: int, frame: np.ndarray, camera_parameters: ocv.CameraParams):
        if mark not in self.individual_markers:
            raise ValueError(f'The individual marker {marker.marker_ID_to_str(mark)} is not known')
        detect_tuple = self._get_detector_cache(mark.aruco_dict_id, frame_idx, frame, camera_parameters)
        if not detect_tuple[1] or detect_tuple[1]['ids'] is None or mark.m_id not in detect_tuple[1]['ids']:
            return None, None
        return self._detectors[mark.aruco_dict_id].get_individual_marker_points(mark.m_id, detect_tuple)

    def _get_detector_cache(self, aruco_dict_id: int, frame_idx: int, frame: np.ndarray, camera_parameters: ocv.CameraParams):
        if aruco_dict_id not in self._det_cache or self._det_cache[aruco_dict_id][0]!=frame_idx:
            if frame is None:
                return None
            detect_tuple = self._detectors[aruco_dict_id].detect_markers(frame, camera_parameters)
            self._det_cache[aruco_dict_id] = (frame_idx, detect_tuple)
        return self._det_cache[aruco_dict_id][1]

    def _visualize_plane(self, plane_name: str, frame_idx: int, frame: np.ndarray):
        if plane_name not in self._plane_to_detector:
            raise ValueError(f'The plane {plane_name} is not known')
        aruco_dict_id = self._plane_to_detector[plane_name]
        if aruco_dict_id in self._last_viz_frame_idx and self._last_viz_frame_idx[aruco_dict_id]==frame_idx:
            # nothing to do, already drawn
            return
        detect_tuple = self._get_detector_cache(aruco_dict_id, frame_idx, None, None)
        if detect_tuple is not None:
            frame = self._detectors[aruco_dict_id].visualize(frame, detect_tuple, plane_marker_color=self._plane_marker_color, recovered_plane_marker_color=self._recovered_plane_marker_color, individual_marker_color=self._individual_marker_color, unexpected_marker_color=self._unexpected_marker_color, rejected_marker_color=self._rejected_marker_color)
            self._last_viz_frame_idx[aruco_dict_id] = frame_idx

    def _visualize_individual_marker(self, mark: marker.MarkerID, frame_idx: int, frame: np.ndarray):
        if mark not in self.individual_markers:
            raise ValueError(f'The individual marker {marker.marker_ID_to_str(mark)} is not known')
        if mark.aruco_dict_id in self._last_viz_frame_idx and self._last_viz_frame_idx[mark.aruco_dict_id]==frame_idx:
            # nothing to do, already drawn
            return
        detect_tuple = self._get_detector_cache(mark.aruco_dict_id, frame_idx, None, None)
        if detect_tuple is not None:
            frame = self._detectors[mark.aruco_dict_id].visualize(frame, detect_tuple, plane_marker_color=self._plane_marker_color, recovered_plane_marker_color=self._recovered_plane_marker_color, individual_marker_color=self._individual_marker_color, unexpected_marker_color=self._unexpected_marker_color, rejected_marker_color=self._rejected_marker_color)
            self._last_viz_frame_idx[mark.aruco_dict_id] = frame_idx


def create_board(board_corner_points: list[np.ndarray], ids: list[int], ArUco_dict: cv2.aruco.Dictionary):
    board_corner_points = np.dstack(board_corner_points)        # list of 2D arrays -> 3D array
    board_corner_points = np.rollaxis(board_corner_points,-1)   # 4x2xN -> Nx4x2
    board_corner_points = np.pad(board_corner_points,((0,0),(0,0),(0,1)),'constant', constant_values=(0.,0.)) # Nx4x2 -> Nx4x3 (at Z=0 to all points)
    return cv2.aruco.Board(board_corner_points, ArUco_dict, np.array(ids))