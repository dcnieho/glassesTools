import numpy as np
import cv2
import pathlib
import typing
import itertools
import math
import copy
from dataclasses import dataclass

from glassesTools.utils import freeze

from . import annotation, drawing, marker, ocv, plane, pose, transforms

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

def parameter_defaults(which: str) -> dict:
    cls = {'detector': cv2.aruco.DetectorParameters, 'refine': cv2.aruco.RefineParameters}[which]
    params = cls()
    values = {name: getattr(params, name) for name in dir(params)
                if not name.startswith('_') and type(getattr(params, name)) in (bool, int, float)}
    # override default corner refinement method to subpixel
    if which == 'detector':
        values['cornerRefinementMethod'] = cv2.aruco.CORNER_REFINE_SUBPIX
    return values

def settings_defaults() -> dict:
    return dict(detector_params=parameter_defaults('detector'),
                refine_params=parameter_defaults('refine'),
                refine=True,
                undistort=False,
                pose_consistency=dict(
                    enabled=False,
                    max_reprojection_error_fraction=.10,
                    min_reprojection_error_px=2.,
                    max_reprojection_error_px=6.,
                    fast_path_min_spread_fraction=.05,
                    max_subsets=500,
                    max_marker_tilt_angle_deg=80.,
                    temporal_enabled=True,
                    temporal_history_size=5,
                    max_translation_speed_m_s=3.,
                    max_rotation_speed_deg_s=400.)
                )

corner_refinement_methods = typing.Literal[
        cv2.aruco.CORNER_REFINE_NONE, cv2.aruco.CORNER_REFINE_SUBPIX,
        cv2.aruco.CORNER_REFINE_CONTOUR, cv2.aruco.CORNER_REFINE_APRILTAG]

# Parameter descriptions checked against OpenCV 4.13.0:
# https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html
# https://docs.opencv.org/4.13.0/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
# https://docs.opencv.org/4.13.0/d5/d09/structcv_1_1aruco_1_1RefineParameters.html
# Implementation details (especially recovery limits and AprilTag preprocessing):
# https://github.com/opencv/opencv/blob/4.13.0/modules/objdetect/src/aruco/aruco_detector.cpp
# https://github.com/opencv/opencv/blob/4.13.0/modules/objdetect/src/aruco/apriltag/apriltag_quad_thresh.cpp
corner_refinement_doc: dict[int, tuple[str, str]] = {
    cv2.aruco.CORNER_REFINE_NONE: (
        'None',
        'Use the initially detected corners without further refinement (CORNER_REFINE_NONE).',
    ),
    cv2.aruco.CORNER_REFINE_SUBPIX: (
        'Subpixel',
        'Refine corner positions to subpixel precision (CORNER_REFINE_SUBPIX).',
    ),
    cv2.aruco.CORNER_REFINE_CONTOUR: (
        'Contour',
        'Refine corners by fitting lines to contour points (CORNER_REFINE_CONTOUR).',
    ),
    cv2.aruco.CORNER_REFINE_APRILTAG: (
        'AprilTag',
        'Use AprilTag quadrilateral detection (CORNER_REFINE_APRILTAG).',
    ),
}

detector_parameter_doc: dict[str, tuple[str, str]] = {
    'adaptiveThreshConstant': (
        'Adaptive threshold constant',
        'Offset subtracted from the local mean when thresholding the image for ArUco contours. '
        'Measured in grayscale intensity levels; changes which pixels become foreground.',
    ),
    'adaptiveThreshWinSizeMin': (
        'Minimum adaptive threshold window size',
        'Smallest adaptive-threshold neighborhood, in pixels. OpenCV tries windows from this value '
        'through adaptiveThreshWinSizeMax, in adaptiveThreshWinSizeStep increments. Even sizes are increased to the next odd size.',
    ),
    'adaptiveThreshWinSizeMax': (
        'Maximum adaptive threshold window size',
        'Upper limit of the adaptive-threshold window-size sweep, in pixels. '
        'Larger windows can preserve the borders of large markers; additional sizes cost processing time.',
    ),
    'adaptiveThreshWinSizeStep': (
        'Adaptive threshold window step',
        'Pixel increment between successive adaptive-threshold window sizes. '
        'Smaller steps test more scales and take more time.',
    ),
    'aprilTagCriticalRad': (
        'AprilTag critical corner angle',
        'AprilTag corner-angle exclusion margin, in radians. Rejects adjoining edges with angles '
        'near 0 or pi; 0 disables this angle test.',
    ),
    'aprilTagDeglitch': (
        'AprilTag noise cleanup',
        'AprilTag binary-image cleanup: 0 disables it; nonzero enables a dilation/erosion pass intended for noisy images.',
    ),
    'aprilTagMaxLineFitMse': (
        'AprilTag maximum line fitting error',
        'Largest permitted line-fit mean squared error for AprilTag edges, in quad-image pixels squared. '
        'Lower values demand straighter edges.',
    ),
    'aprilTagMaxNmaxima': (
        'AprilTag maximum corner candidates',
        'Maximum number of potential corners considered when fitting an AprilTag quadrilateral. '
        'Larger values allow more combinations to be tested.',
    ),
    'aprilTagMinClusterPixels': (
        'AprilTag minimum cluster size',
        'Minimum number of boundary samples in an AprilTag candidate cluster. Smaller clusters are discarded.',
    ),
    'aprilTagMinWhiteBlackDiff': (
        'AprilTag minimum contrast',
        'Minimum local brightness contrast for AprilTag thresholding, in 8-bit grayscale levels. '
        'Lower values allow weaker contrast.',
    ),
    'aprilTagQuadDecimate': (
        'AprilTag downsampling factor',
        'Downsampling factor for AprilTag quad extraction. Values above 1 reduce each image dimension '
        'by this factor; 0 and 1 retain the input resolution. Decoding still uses the full-resolution input.',
    ),
    'aprilTagQuadSigma': (
        'AprilTag smoothing sigma',
        'Gaussian smoothing sigma, in pixels, before AprilTag quad extraction. 0 disables filtering; '
        'larger positive values suppress noise but can blur small markers.',
    ),
    'cornerRefinementMaxIterations': (
        'Maximum corner refinement iterations',
        'Iteration limit for subpixel corner optimization. The accuracy threshold can terminate it earlier.',
    ),
    'cornerRefinementMethod': (
        'Corner refinement method',
        'Method used for refining detected marker corners.',
    ),
    'cornerRefinementMinAccuracy': (
        'Corner refinement accuracy',
        'Subpixel refinement stops when a corner moves less than this distance between iterations, '
        'in pixels. Smaller values demand tighter convergence.',
    ),
    'cornerRefinementWinSize': (
        'Corner refinement window size',
        'Maximum subpixel search half-window, in pixels: 5 permits an 11 by 11 neighborhood. '
        'The window can shrink for small markers according to relativeCornerRefinmentWinSize.',
    ),
    'detectInvertedMarker': (
        'Detect inverted markers',
        'Also accept markers with reversed black/white polarity. This concerns image intensity, '
        'not mirrored patterns or viewing the printed marker from behind.',
    ),
    'errorCorrectionRate': (
        'Error correction rate',
        'Fraction of the dictionary bit-error correction capacity used during initial decoding, from 0 to 1. '
        'Higher values tolerate more bit errors but can increase misidentification.',
    ),
    'markerBorderBits': (
        'Marker border bits',
        'Black border width measured in marker cells, not image pixels. Must match the printed marker.',
    ),
    'maxErroneousBitsInBorderRate': (
        'Maximum border error rate',
        'Allowance for incorrectly white cells in the black border. OpenCV multiplies this rate by '
        'the number of inner code cells to obtain the allowed error count.',
    ),
    'maxMarkerPerimeterRate': (
        'Maximum marker perimeter ratio',
        'Upper candidate-contour perimeter limit, expressed as a multiple of the larger image dimension. '
        'Candidates above this size are discarded.',
    ),
    'minCornerDistanceRate': (
        'Minimum corner separation ratio',
        'Required separation of adjacent candidate corners, expressed as a fraction of the contour perimeter. '
        'Rejects quadrilaterals with corners too close together.',
    ),
    'minDistanceToBorder': (
        'Minimum distance to image edge',
        'Required clearance from every marker corner to the image edges, in pixels. '
        'Rejects candidates too close to an edge.',
    ),
    'minGroupDistance': (
        'Minimum distance within groups',
        'Within a group of nearby contours, minimum corner separation for keeping additional candidates, '
        'relative to marker-cell size. Larger values retain fewer close alternatives.',
    ),
    'minMarkerDistanceRate': (
        'Minimum marker distance ratio',
        'Corner-distance threshold for grouping nearby marker candidates, relative to the smaller contour perimeter. '
        'Larger values group more candidates together.',
    ),
    'minMarkerLengthRatioOriginalImg': (
        'Minimum marker length ratio',
        'ArUco3 scale-selection parameter relative to the larger input-image dimension. '
        'Higher values increase downsampling and speed, at the cost of detecting small markers; 0 avoids this downsampling.',
    ),
    'minMarkerPerimeterRate': (
        'Minimum marker perimeter ratio',
        'Lower candidate-contour perimeter limit, expressed as a fraction of the larger image dimension. '
        'Smaller values admit smaller markers and more noise candidates.',
    ),
    'minOtsuStdDev': (
        'Minimum standard deviation for Otsu',
        'Minimum grayscale standard deviation for Otsu thresholding during bit decoding. '
        'Below this contrast, all cells are assigned one value based on the patch mean.',
    ),
    'minSideLengthCanonicalImg': (
        'Minimum canonical marker size',
        'ArUco3 minimum marker size in the canonical working image, in pixels per side. '
        'Used with minMarkerLengthRatioOriginalImg to select processing scales.',
    ),
    'perspectiveRemoveIgnoredMarginPerCell': (
        'Ignored margin per decoded cell',
        'Fraction of each rectified cell width excluded on each edge when reading its bit. '
        'Larger margins avoid boundary contamination but leave fewer pixels for voting.',
    ),
    'perspectiveRemovePixelPerCell': (
        'Pixels per decoded cell side',
        'Pixels per cell side in the perspective-corrected decoding image, including border cells. '
        'Higher values sample each bit more densely.',
    ),
    'polygonalApproxAccuracyRate': (
        'Polygon approximation tolerance',
        'Polygon-approximation tolerance as a fraction of candidate contour length. '
        'Controls how closely the contour must resemble a four-corner polygon.',
    ),
    'relativeCornerRefinmentWinSize': (
        'Relative corner refinement window size',
        'Subpixel search half-window relative to the average marker-cell size. '
        'OpenCV rounds it to pixels, with a minimum of 1 and a cap of cornerRefinementWinSize.',
    ),
    'useAruco3Detection': (
        'Use ArUco3 detection',
        'Enable accelerated ArUco3 detection using image scales. Requires subpixel corner refinement. '
        'The canonical-size and marker-length-ratio parameters control the size/speed tradeoff.',
    ),
}

refine_parameter_doc: dict[str, tuple[str, str]] = {
    'checkAllOrders': (
        'Check all corner orders',
        'Try all four cyclic corner orders when matching a rejected quadrilateral to a missing board marker. '
        'If disabled, use only its supplied corner order.',
    ),
    'errorCorrectionRate': (
        'Recovery error correction rate',
        'Board-recovery bit-error allowance relative to the dictionary correction capacity. '
        'Values above 1 allow looser recovery; -1 skips code checking. '
        'Setting this to zero effectively accepts no candidates.',
    ),
    'minRepDistance': (
        'Recovery distance tolerance',
        'Pixel tolerance for matching rejected candidates to predicted board markers, '
        'measured using the largest corresponding-corner discrepancy. Larger values allow recovery farther from the prediction.',
    ),
}

def settings_problems(settings: dict | None = None) -> dict[tuple[str, ...], str]:
    return _resolve_settings(settings)[1]

def resolve_settings(settings: dict | None = None) -> dict:
    out, problems = _resolve_settings(settings)
    if problems:
        raise ValueError('\n'.join(dict.fromkeys(problems.values())))
    return out

def _resolve_settings(settings: dict | None) -> tuple[dict, dict[tuple[str, ...], str]]:
    out = settings_defaults()
    problems = {}
    if settings is not None and not isinstance(settings, dict):
        return out, {(): 'ArUco settings must be a dictionary'}

    for key, value in (settings or {}).items():
        if key not in out:
            problems[(key,)] = f'Unknown ArUco setting: {key}'
            continue
        if isinstance(out[key], dict):
            cls = {'detector_params': cv2.aruco.DetectorParameters,
                   'refine_params': cv2.aruco.RefineParameters}.get(key)
            if not isinstance(value, dict):
                problems[(key,)] = f'The value of the ArUco setting "{key}" must be a dictionary'
                continue
            for name, val in value.items():
                path = (key, name)
                if name not in out[key]:
                    problems[path] = (f'{name} is not a valid parameter for cv2.aruco.{cls.__name__}'
                                      if cls is not None else f'Unknown ArUco {key} parameter: {name}')
                    continue
                default = out[key][name]
                if type(default) is float and type(val) in (int, float):
                    try:
                        val = float(val)
                    except OverflowError:
                        problems[path] = f'ArUco {key}.{name} must be a finite float'
                        continue
                if type(val) is not type(default) or (type(val) is float and not math.isfinite(val)):
                    problems[path] = f'ArUco {key}.{name} must be a finite {type(default).__name__}'
                    continue
                out[key][name] = val
        elif type(value) is not bool:
            problems[(key,)] = f'ArUco {key} must be a boolean'
        else:
            out[key] = value

    def check(group, names, valid, message):
        paths = [(group, name) for name in names]
        # Type errors and earlier constraints take precedence over dependent checks.
        if not valid and not any(path in problems for path in paths):
            for path in paths:
                problems[path] = message

    d, r, c = out['detector_params'], out['refine_params'], out['pose_consistency']
    positive = ('adaptiveThreshWinSizeMin', 'adaptiveThreshWinSizeMax', 'adaptiveThreshWinSizeStep',
                'minMarkerPerimeterRate', 'maxMarkerPerimeterRate', 'polygonalApproxAccuracyRate',
                'cornerRefinementWinSize', 'cornerRefinementMaxIterations', 'cornerRefinementMinAccuracy',
                'markerBorderBits', 'perspectiveRemovePixelPerCell')
    for name in positive:
        check('detector_params', [name], d[name] > 0, f'ArUco detector_params.{name} must be positive')
    check('detector_params', ['adaptiveThreshWinSizeMin'], d['adaptiveThreshWinSizeMin'] >= 3,
          'adaptiveThreshWinSizeMin must be at least 3')
    check('detector_params', ['adaptiveThreshWinSizeMin', 'adaptiveThreshWinSizeMax'],
          d['adaptiveThreshWinSizeMax'] >= d['adaptiveThreshWinSizeMin'],
          'adaptiveThreshWinSizeMax must be at least adaptiveThreshWinSizeMin')
    check('detector_params', ['minMarkerPerimeterRate', 'maxMarkerPerimeterRate'],
          d['maxMarkerPerimeterRate'] >= d['minMarkerPerimeterRate'],
          'maxMarkerPerimeterRate must be at least minMarkerPerimeterRate')
    check('detector_params', ['cornerRefinementMethod'], d['cornerRefinementMethod'] in typing.get_args(corner_refinement_methods),
          'Unknown ArUco cornerRefinementMethod')
    check('detector_params', ['useAruco3Detection', 'cornerRefinementMethod'],
          not d['useAruco3Detection'] or d['cornerRefinementMethod'] == cv2.aruco.CORNER_REFINE_SUBPIX,
          'OpenCV ArUco3 detection requires CORNER_REFINE_SUBPIX')
    for name in ('minCornerDistanceRate', 'minDistanceToBorder', 'minMarkerDistanceRate', 'minGroupDistance',
                 'minOtsuStdDev', 'minSideLengthCanonicalImg', 'minMarkerLengthRatioOriginalImg',
                 'relativeCornerRefinmentWinSize', 'aprilTagQuadDecimate', 'aprilTagQuadSigma'):
        check('detector_params', [name], d[name] >= 0, f'ArUco detector_params.{name} cannot be negative')
    for name in ('errorCorrectionRate', 'maxErroneousBitsInBorderRate'):
        check('detector_params', [name], 0 <= d[name] <= 1, f'ArUco detector_params.{name} must be between 0 and 1')
    check('detector_params', ['perspectiveRemoveIgnoredMarginPerCell'], 0 <= d['perspectiveRemoveIgnoredMarginPerCell'] < .5,
          'ArUco perspectiveRemoveIgnoredMarginPerCell must be in [0, 0.5)')
    check('detector_params', ['useAruco3Detection', 'minSideLengthCanonicalImg', 'minMarkerLengthRatioOriginalImg'],
          not d['useAruco3Detection'] or bool(d['minSideLengthCanonicalImg'] or d['minMarkerLengthRatioOriginalImg']),
          'ArUco3 requires a nonzero minimum marker size')
    check('refine_params', ['minRepDistance'], r['minRepDistance'] > 0, 'ArUco refine_params.minRepDistance must be positive')
    check('pose_consistency', ['max_reprojection_error_fraction'], 0 < c['max_reprojection_error_fraction'] <= 1,
          'ArUco pose consistency max_reprojection_error_fraction must be in (0, 1]')
    check('pose_consistency', ['min_reprojection_error_px'], c['min_reprojection_error_px'] > 0,
          'ArUco pose consistency min_reprojection_error_px must be positive')
    check('pose_consistency', ['max_reprojection_error_px', 'min_reprojection_error_px'],
          c['max_reprojection_error_px'] >= c['min_reprojection_error_px'],
          'ArUco pose consistency max_reprojection_error_px must be at least min_reprojection_error_px')
    check('pose_consistency', ['fast_path_min_spread_fraction'],
          0 <= c['fast_path_min_spread_fraction'] <= 1,
          'ArUco pose consistency fast_path_min_spread_fraction must be in [0, 1]')
    check('pose_consistency', ['max_subsets'], type(c['max_subsets']) is int and c['max_subsets'] >= 1,
          'ArUco pose consistency max_subsets must be a positive integer')
    check('pose_consistency', ['max_marker_tilt_angle_deg'], 0 <= c['max_marker_tilt_angle_deg'] < 90,
          'ArUco pose consistency max_marker_tilt_angle_deg must be in [0, 90) degrees')
    check('pose_consistency', ['temporal_history_size'],
          type(c['temporal_history_size']) is int and c['temporal_history_size'] >= 1,
          'ArUco pose consistency temporal_history_size must be a positive integer')
    check('pose_consistency', ['max_translation_speed_m_s'], c['max_translation_speed_m_s'] > 0,
          'ArUco pose consistency max_translation_speed_m_s must be positive')
    check('pose_consistency', ['max_rotation_speed_deg_s'], c['max_rotation_speed_deg_s'] > 0,
          'ArUco pose consistency max_rotation_speed_deg_s must be positive')

    # Round-trip valid parameters so comparisons use OpenCV's actual precision (float32, not double in case of floats).
    for which in ('detector', 'refine'):
        obj = cv2.aruco.DetectorParameters() if which == 'detector' else cv2.aruco.RefineParameters()
        key = f'{which}_params'
        for name, value in out[key].items():
            if (key, name) in problems:
                continue
            try:
                setattr(obj, name, value)
            except (TypeError, ValueError, OverflowError, cv2.error) as exc:
                problems[(key, name)] = f'ArUco {key}.{name}: {exc}'
        out[key] = {name: getattr(obj, name) for name in out[key]}
    return out, problems


def _project_points(object_points, rvec, tvec, frame_info, camera_params):
    return transforms.project_points(np.asarray(object_points).reshape(-1, 3), camera_params, rot_vec=rvec, trans_vec=tvec, ROI_offset=[frame_info.get('offset_x', 0), frame_info.get('offset_y', 0)])

def _residuals(object_points, image_points, rvec, tvec, frame_info, camera_params, shape):
    projected_points = _project_points(object_points, rvec, tvec, frame_info, camera_params)
    return np.asarray(projected_points).reshape(shape) - np.asarray(image_points).reshape(shape)

def _reprojection_error_distance(object_points, image_points, rvec, tvec, frame_info, camera_params):
    residuals = _residuals(object_points, image_points, rvec, tvec, frame_info, camera_params, (-1, 4, 2))
    return np.linalg.norm(residuals, axis=2).max(axis=1)

def _reprojection_error_rms(object_points, image_points, rvec, tvec, frame_info, camera_params):
    residuals = _residuals(object_points, image_points, rvec, tvec, frame_info, camera_params, (-1, 2))
    return float(np.sqrt(np.mean(residuals ** 2)))

def _pose_candidates(object_points, image_points, frame_info, camera_params):
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(obj) < 4 or not np.isfinite(obj).all() or not np.isfinite(img).all():
        return []
    offset = np.array([frame_info.get('offset_x', 0), frame_info.get('offset_y', 0)])
    if camera_params.has_opencv_camera():
        solve_img, K, D = img + offset, camera_params.camera_mtx, camera_params.distort_coeffs
    elif camera_params.has_colmap_camera():
        rays = transforms.unproject_points(img, camera_params, offset)
        solve_img, K, D = rays[:, :2] / rays[:, 2:], np.eye(3), np.zeros(5)
    else:
        raise ValueError('ArUco pose consistency checking requires camera calibration')
    # determine if all markers are coplanar
    centered = obj - obj.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[1] <= max(singular[0], 1.) * 1.e-10:
        return []
    planar = singular[2] <= singular[0] * 1.e-6
    # Use IPPE and iterative PnP for planar markers, SQPNP for nonplanar markers.
    flags = [cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE] if planar else [cv2.SOLVEPNP_SQPNP]
    solutions = []
    for flag in flags:
        try:
            count, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, solve_img.reshape(-1, 1, 2), K, D, flags=flag)
        except cv2.error:
            continue
        for rvec, tvec in zip(rvecs, tvecs):
            if np.isfinite(rvec).all() and np.isfinite(tvec).all():
                rms_reprojection_error = _reprojection_error_rms(obj, img, rvec, tvec, frame_info, camera_params)
                solutions.append((len(obj), rvec, tvec, rms_reprojection_error))
    return solutions

def _marker_errors(object_corners, image_corners, rvec, tvec, frame_info, camera_params, min_facing_cosine):
    obj = np.asarray(object_corners, dtype=np.float64).reshape(-1, 4, 3)
    img = np.asarray(image_corners, dtype=np.float64).reshape(-1, 4, 2)
    # check orientation of marker relative to camera: if the marker is facing away or too tilted, it is rejected
    R = cv2.Rodrigues(rvec)[0]
    cam = obj @ R.T + np.asarray(tvec).reshape(1, 1, 3)
    # ArUco corners are TL, TR, BR, BL as viewed from the printed front.
    normals = np.cross(cam[:, 3] - cam[:, 0], cam[:, 1] - cam[:, 0])
    to_camera = -cam.mean(axis=1)
    denom = np.linalg.norm(normals, axis=1) * np.linalg.norm(to_camera, axis=1)
    facing = np.einsum('ij,ij->i', normals, to_camera) / np.maximum(denom, np.finfo(float).tiny)
    valid = (cam[:, :, 2] > 0).all(axis=1) & (denom > 0) & (facing >= min_facing_cosine)
    # Get error for worst corner. This enforces that every corner must agree: partial-marker inliers are deliberately not accepted.
    errors = _reprojection_error_distance(obj, img, rvec, tvec, frame_info, camera_params)
    return np.where(valid & np.isfinite(errors), errors, np.inf)

def _marker_reprojection_limits(image_corners, settings):
    # determine the maximum reprojection error for each marker based on its size in the image, with limits
    img = np.asarray(image_corners, dtype=np.float64).reshape(-1, 4, 2)
    edge_lengths = np.linalg.norm(np.roll(img, -1, axis=1) - img, axis=2)
    marker_sizes = np.median(edge_lengths, axis=1)
    return np.clip(settings['max_reprojection_error_fraction'] * marker_sizes, settings['min_reprojection_error_px'], settings['max_reprojection_error_px'])

def _marker_spread_fraction(selected_centers, layout_centers):
    # determine the fraction of the board layout that is covered by the selected markers, in each axis
    all_span = np.ptp(layout_centers, axis=0)
    active_axes = all_span > np.finfo(float).eps
    if not active_axes.any():
        return 0.
    selected_span = np.ptp(selected_centers, axis=0)
    ratios = np.clip(selected_span[active_axes] / all_span[active_axes], 0., 1.)
    return float(np.prod(ratios))

def _spatially_spread_subsets(groups, centers, subset_size, max_subsets=None):
    # generate subsets of groups that are spatially spread out, up to the requested size
    # a group is a list of indices of detections that share the same marker ID (one detection per marker is selected for each subset). N multiple detections are expected to contain N-1 false positives, since markers on our boards are unique.
    if max_subsets is not None and max_subsets <= 0:
        return
    count = len(groups)
    size = min(count, max(1, subset_size))
    group_subsets = set()
    for start in range(count):
        chosen = [start]
        while len(chosen) < size:
            remaining = (i for i in range(count) if i not in chosen)
            next_group = max(remaining, key=lambda i: (min(float(np.sum((centers[i] - centers[j]) ** 2)) for j in chosen), -i))
            chosen.append(next_group)
        group_subsets.add(tuple(sorted(chosen)))
    ordered = sorted(group_subsets, key=lambda subset: (-_marker_spread_fraction(centers[list(subset)], centers), subset))
    yielded = 0
    for subset in ordered:
        if max_subsets is not None and yielded >= max_subsets:
            return
        base = [groups[group][0] for group in subset]
        yield base
        yielded += 1
        for position, group in enumerate(subset):
            for alternative in groups[group][1:]:
                if max_subsets is not None and yielded >= max_subsets:
                    return
                variant = base.copy()
                variant[position] = alternative
                yield variant
                yielded += 1
    for group in range(count):
        for detection in groups[group]:
            if max_subsets is not None and yielded >= max_subsets:
                return
            yield [detection]
            yielded += 1

def _rotation_distance_deg(rotation_a, rotation_b):
    # get overall rotation difference in degrees between two rotation matrices
    relative = rotation_a @ rotation_b.T
    cosine = np.clip((np.trace(relative) - 1.) / 2., -1., 1.)
    return math.degrees(math.acos(float(cosine)))

def _pose_history_entry(rvec, tvec, timestamp_ms, unit_to_meters):
    rotation = cv2.Rodrigues(rvec)[0]
    position_m = -rotation.T @ np.asarray(tvec).reshape(3) * unit_to_meters
    return float(timestamp_ms), position_m, rotation

def _temporal_pose_score(rvec, tvec, timestamp_ms, history, unit_to_meters, settings):
    # compute a score based on the median translation and rotation speeds over the recent pose history, normalized by the maximum allowed speeds
    if not settings['temporal_enabled'] or timestamp_ms is None or not history:
        return 0.
    current = _pose_history_entry(rvec, tvec, timestamp_ms, unit_to_meters)
    recent = history[-settings['temporal_history_size']:]
    translation_speeds, rotation_speeds = [], []
    for previous in recent:
        elapsed = (current[0] - previous[0]) / 1000.
        if elapsed <= 0:
            continue
        translation_speeds.append(float(np.linalg.norm(current[1] - previous[1])) / elapsed)
        rotation_speeds   .append(_rotation_distance_deg(current[2], previous[2]) / elapsed)
    if not translation_speeds:
        return 0.
    return max(float(np.median(translation_speeds)) / settings['max_translation_speed_m_s'],
               float(np.median(rotation_speeds))    / settings['max_rotation_speed_deg_s'])

def _consistent_pose(board, corners, ids, frame_info, camera_params, settings, min_markers=3, history=(), unit_to_meters=1.) -> tuple[list[int], tuple[int, None, None, float] | tuple[int, np.ndarray, np.ndarray, float]]:
    # check whether pose is consistent with the board geometry and recent pose history, and physically plausible. In the process, markers that may be misdetected are filtered out.
    # Returns the set of selected markers, and the best pose and its reprojection error.
    # The inner tuple represents the best pose and its reprojection error, and contains:
    #   0: number of corners used in the best pose. It is 0 on failure.
    #   1: rotation vector of the best pose
    #   2: translation vector of the best pose
    #   3: RMS reprojection error of the best pose: how far the pose's projected corners differ from the detected corners, in pixels.
    if not camera_params.has_intrinsics():
        raise ValueError('ArUco pose consistency checking requires camera calibration')
    failed = (0, None, None, -1.)
    if ids is None or not len(ids):
        return [], failed
    # Build board geometry for each detection and group detections by marker ID.
    ids = np.asarray(ids).flatten()
    board_map = {int(m_id): np.asarray(obj) for m_id, obj in zip(board.getIds(), board.getObjPoints())}
    layout_centers = np.asarray([points.mean(axis=0)[:2] for points in board_map.values()])
    obj = np.asarray([board_map[int(mid)] for mid in ids]).reshape(-1, 4, 3)
    centers = obj.mean(axis=1)[:, :2]
    img = np.asarray([np.asarray(c).reshape(4, 2) for c in corners])
    # Group detections by marker ID. groups should mostly have one detection per marker since markers on our boards are unique, but multiple detections are possible in case of false positives.
    groups = {int(m_id): np.flatnonzero(ids == m_id).tolist() for m_id in np.unique(ids)}
    # precompure scoring limits for marker orientation and reprojection error
    min_facing_cosine = math.cos(math.radians(settings['max_marker_tilt_angle_deg']))
    reprojection_limits = _marker_reprojection_limits(img, settings)
    timestamp_ms = frame_info.get('timestamp_ms')
    # init search
    best_indices, best_pose, best_error, best_spread, best_temporal_score = [], failed, np.inf, 0., np.inf

    # Score each pose by marker-level reprojection, orientation, and temporal consistency.
    def score(solution):
        # Get reprojection error for each marker and reject markers that are facing away or too tilted.
        errors = _marker_errors(obj, img, solution[1], solution[2], frame_info, camera_params, min_facing_cosine)
        normalized_errors = errors / reprojection_limits
        selected = [min(indices, key=lambda i: normalized_errors[i]) for indices in groups.values()]
        selected = sorted(i for i in selected if normalized_errors[i] <= 1.)
        # Reject poses that are inconsistent with the recent pose history, or that use too few markers.
        temporal_score = _temporal_pose_score(solution[1], solution[2], timestamp_ms, history, unit_to_meters, settings)
        if len(selected) < min_markers or temporal_score > 1.:
            return [], np.inf, 0., np.inf
        # Compute the spatial spread of the selected markers relative to the board layout. This is used to prefer poses that cover more of the board.
        spread = _marker_spread_fraction(centers[selected], layout_centers)
        mean_normalized_marker_error = float(np.mean(normalized_errors[selected]))
        return (selected, mean_normalized_marker_error, spread, temporal_score)

    # Rank candidates by marker count, board coverage, reprojection error, and motion.
    def record(selected, error, spread, temporal_score, candidate, force=False):
        nonlocal best_indices, best_pose, best_error, best_spread, best_temporal_score
        rank = (len(selected), spread, -error, -temporal_score)
        best_rank = (len(best_indices), best_spread, -best_error, -best_temporal_score)
        if not force and rank <= best_rank:
            return
        best_indices, best_error, best_spread, best_temporal_score = selected, error, spread, temporal_score
        rms = _reprojection_error_rms(obj[selected], img[selected], candidate[1], candidate[2], frame_info, camera_params)
        best_pose = (len(selected) * 4, candidate[1], candidate[2], rms)

    # Fit candidate subsets, then refit each pose using its consensus markers.
    def consider(indices):
        candidates = _pose_candidates(obj[indices], img[indices], frame_info, camera_params)
        for candidate in candidates:
            selected, error, spread, temporal_score = score(candidate)
            if len(selected) < min_markers:
                continue
            # Refit only the consensus, then recheck both geometry and membership.
            for _ in range(3):
                refits = _pose_candidates(obj[selected], img[selected], frame_info, camera_params)
                acceptable = []
                for fitted in refits:
                    next_indices, next_error, next_spread, next_temporal_score = score(fitted)
                    if len(next_indices) >= min_markers:
                        acceptable.append((next_indices, next_error, next_spread, next_temporal_score, fitted))
                if not acceptable:
                    break
                next_indices, next_error, next_spread, next_temporal_score, fitted = min(acceptable, key=lambda x: (-len(x[0]), -x[2], x[1], x[3]))
                unchanged = next_indices == selected
                selected, error, spread, temporal_score, candidate = next_indices, next_error, next_spread, next_temporal_score, fitted
                if unchanged:
                    break
            record(selected, error, spread, temporal_score, candidate)

    # To keep this fast, if all marker detections are unique (so no markers detected multiple times), just fit the pose and check its quality. If the pose is acceptable, return it. If not, then try refitting the pose using only the consensus markers. If that fails, then search all subsets of detections up to a limit.
    fast_candidate = None
    if len(groups) == len(ids): # all markers detected only once
        # this likely returns multiple poses, choose the best
        initial_candidates = _pose_candidates(obj, img, frame_info, camera_params)
        for candidate in initial_candidates:
            # potentially prune markers
            selected, error, spread, temporal_score = score(candidate)
            if len(selected) < min_markers:
                # not enough marker left, not a viable pose
                continue
            sufficient_spread = spread >= settings['fast_path_min_spread_fraction']
            if len(selected) == len(groups) and sufficient_spread:
                # no markers were pruned, and the spread is sufficient, so this is a good candidate
                candidate_rank = (len(selected), spread, -error, -temporal_score)
                # store as best estimate if it is better than any previous candidate
                if fast_candidate is None or candidate_rank > fast_candidate[0]:
                    fast_candidate = (candidate_rank, selected, error, spread, temporal_score, candidate)
                continue
            if len(selected) == len(groups):
                # spread is insufficient, but we can still record it as a candidate in case no better candidates are found
                record(selected, error, spread, temporal_score, candidate)
                continue
            # try refitting the pose using only the markers left after pruning
            record(selected, error, spread, temporal_score, candidate)
            for refitted in _pose_candidates(obj[selected], img[selected], frame_info, camera_params):
                refitted_indices, refitted_error, refitted_spread, refitted_temporal_score = score(refitted)
                if refitted_indices != selected:
                    # further markers were pruned, this indicates we are not converging and should delve into a deeper search. Ignore this pose
                    continue
                record(refitted_indices, refitted_error, refitted_spread, refitted_temporal_score, refitted, force=True)
                if refitted_spread >= settings['fast_path_min_spread_fraction']:
                    # the refitted pose is acceptable and has sufficient spread to be passed, so we can return it without searching all subsets
                    candidate_rank = (len(refitted_indices), refitted_spread, -refitted_error, -refitted_temporal_score)
                    if fast_candidate is None or candidate_rank > fast_candidate[0]:
                        fast_candidate = (candidate_rank, refitted_indices, refitted_error, refitted_spread, refitted_temporal_score, refitted)
        # check if the fast path yielded a good pose
        if fast_candidate is not None:
            # Return the best full or refitted consensus with adequate spread.
            record(*fast_candidate[1:], force=True)
            return best_indices, best_pose

    # Search all subsets when affordable; otherwise sample spatially spread subsets.
    combinations = math.prod(len(g) + 1 for g in groups.values()) - 1
    budget = settings['max_subsets']
    if combinations <= budget:
        choices = itertools.product(*[[-1] + g for g in groups.values()])
        subsets = sorted(([i for i in choice if i != -1] for choice in choices), key=len, reverse=True)
    else:
        group_list = list(groups.values())
        group_centers = np.asarray([centers[indices[0]] for indices in group_list])
        subsets = _spatially_spread_subsets(group_list, group_centers, max(2, min_markers), budget)
    seen = {tuple(range(len(ids)))} if len(groups) == len(ids) else set()
    for subset in subsets:
        if not subset:
            continue
        key = tuple(sorted(subset))
        if key in seen:
            continue
        seen.add(key)
        consider(list(key))
        if (len(best_indices) == len(groups) and best_spread >= settings['fast_path_min_spread_fraction']):
            break
    return best_indices, best_pose


class PlaneSetup(typing.TypedDict):
    plane                   : plane.Plane
    min_num_markers         : int
    aruco_settings          : typing.NotRequired[dict[str, typing.Any]]
class MarkerSetup(typing.TypedDict):
    detect_only             : bool
    size                    : float
    detector_params         : typing.NotRequired[dict[str, typing.Any]]

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
            cv2.imwrite(pathlib.Path(output_dir) / f"{m_id}.png", marker_image)

class Detector:
    def __init__(self, dictionary_id: int, settings: dict | None = None):
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
        self._pose_history          : dict[str, list[tuple]]    = {}
        self._plane_unit_to_meters  : dict[str, float]          = {}

        self.settings                                           = resolve_settings(settings)

        self._det: cv2.aruco.ArucoDetector|None                 = None

        self._last_detect_output : tuple[dict[str,dict[str]],dict[str],dict[str],list[np.ndarray]] = {}

    def add_plane(self, name: str, setup: PlaneSetup):
        self._check_dict(setup['plane'].aruco_dict_id, 'plane')
        unit = (setup['plane'].unit or '').strip().lower()
        unit_to_meters = {'m': 1., 'cm': .01, 'mm': .001}.get(unit)
        consistency = self.settings['pose_consistency']
        if consistency['enabled'] and consistency['temporal_enabled'] and unit_to_meters is None:
            raise ValueError('Temporal ArUco pose consistency requires the plane unit to be m, cm, or mm')
        self._plane_unit_to_meters[name] = unit_to_meters or 1.
        self.planes[name] = setup
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

    def create_detector(self):
        # set detector parameters in an OpenCV settings object
        detector_params = cv2.aruco.DetectorParameters()
        refine_params   = cv2.aruco.RefineParameters()
        for name, value in self.settings['detector_params'].items():
            setattr(detector_params, name, value)
        for name, value in self.settings['refine_params'].items():
            setattr(refine_params, name, value)
        # create a detector with the requested settings
        self._det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(self.dictionary_id), detector_params, refine_params)

    def detect_markers(self, image: cv2.UMat, frame_info: dict, camera_params: ocv.CameraParams, raw_detection=None) -> tuple:
        img_points, ids, rejected = raw_detection if raw_detection is not None else self._detect_markers(image)
        rejected = tuple(rejected)
        out_planes = {}
        consistency = self.settings['pose_consistency']
        pose_cache = {}
        for p in self.planes:
            # get the detections for this plane (that is, filter on expected marker IDs)
            pl_img_points, pl_ids = filter_detections(img_points, ids, self._plane_marker_ids[p])
            if pl_ids is None or not len(pl_ids):
                out_planes[p] = None
                continue

            timestamp_ms = frame_info.get('timestamp_ms')
            history = self._pose_history.setdefault(p, [])
            try:
                timestamp_ms = float(timestamp_ms)
            except (TypeError, ValueError):
                timestamp_ms = None
            if timestamp_ms is None or not math.isfinite(timestamp_ms) or (history and timestamp_ms <= history[-1][0]):
                history.clear()
                timestamp_ms = None if timestamp_ms is None or not math.isfinite(timestamp_ms) else timestamp_ms
            accepted_pose = None

            # If consistency checking is enabled, we will filter out any detections that do not yield a consistent pose with the other detections. This also filters out duplicate detections.
            # If consistency checking is disabled, we will filter out any duplicate detections of the same marker.
            if consistency['enabled']:
                selected, accepted_pose = self._consistent_pose(self._boards[p], pl_img_points, pl_ids, frame_info, camera_params, self.planes[p]['min_num_markers'], pose_cache, history, self._plane_unit_to_meters[p])
                rejected += tuple(c for i, c in enumerate(pl_img_points) if i not in selected)
                pl_img_points, pl_ids = [pl_img_points[i] for i in selected], pl_ids[selected]
            else:
                ok, kept, kept_ids, rejected_indices = filter_board_duplicates(self._boards[p], pl_img_points, pl_ids, frame_info, camera_params)
                if ok:
                    rejected += tuple(pl_img_points[i] for i in rejected_indices)
                    pl_img_points, pl_ids = kept, kept_ids
            recovered_ids = None

            # Preserve the existing recovery threshold independently of the on/off switch.
            if self.settings['refine'] and len(pl_ids) >= self.planes[p]['min_num_markers']:
                pl_img_points, pl_ids, rejected, recovered_ids = self._refine_detection(image, pl_img_points, pl_ids, rejected, self._boards[p], frame_info, camera_params)
                rejected = tuple(rejected)

                # Validate the refined set, including newly recovered markers if any
                if consistency['enabled']:
                    selected, accepted_pose = self._consistent_pose(self._boards[p], pl_img_points, pl_ids, frame_info, camera_params, self.planes[p]['min_num_markers'], pose_cache, history, self._plane_unit_to_meters[p])
                    rejected += tuple(c for i, c in enumerate(pl_img_points) if i not in selected)
                    pl_img_points, pl_ids = [pl_img_points[i] for i in selected], pl_ids[selected]
                    if recovered_ids is not None:
                        recovered_ids = recovered_ids[np.isin(recovered_ids.flatten(), pl_ids.flatten())]

            if (consistency['enabled'] and consistency['temporal_enabled'] and timestamp_ms is not None and accepted_pose is not None and accepted_pose[1] is not None):
                history.append(_pose_history_entry(accepted_pose[1], accepted_pose[2], timestamp_ms, self._plane_unit_to_meters[p]))
                del history[:-consistency['temporal_history_size']]

            out_planes[p] = dict(img_points=pl_img_points, ids=pl_ids, recovered_ids=recovered_ids)
        out_individual = dict(zip(('img_points', 'ids'), filter_detections(img_points, ids, self._individual_marker_ids)))
        unexpected = dict(zip(('img_points', 'ids'), filter_detections(img_points, ids, self._all_markers, keep_expected=False)))
        self._last_detect_output = (out_planes, out_individual, unexpected, rejected)
        return self._last_detect_output

    def _consistent_pose(self, board, corners, ids, frame_info, camera_params, min_num_markers, cache, history=(), unit_to_meters=1.):
        key = freeze((board.getObjPoints(), board.getIds(), corners, ids, frame_info.get('timestamp_ms'), history, self.settings['pose_consistency'], min_num_markers, unit_to_meters, self.settings['undistort']))
        if key not in cache:
            cache[key] = _consistent_pose(board, corners, ids, frame_info, camera_params, self.settings['pose_consistency'], min_num_markers, history, unit_to_meters)
        return copy.deepcopy(cache[key])

    def _detect_markers(self, image: cv2.UMat):
        img_points, ids, rejected_img_points = self._det.detectMarkers(image)
        if np.any(ids==None):
            ids = None
        return img_points, ids, rejected_img_points

    def _refine_detection(self, image: cv2.UMat, detected_corners, detected_ids, rejected_corners, board: cv2.aruco.Board, frame_info: dict, camera_params: ocv.CameraParams):
        return refine_detection(image, detected_corners, detected_ids, rejected_corners, self._det, board, frame_info, camera_params)

    def _filter_detections(self, img_points: list[np.ndarray], ids: np.ndarray, expected_ids: list[np.ndarray], keep_expected=True):
        return filter_detections(img_points, ids, expected_ids, keep_expected)

    def get_matching_image_board_points(self, plane_name: str, detect_tuple=None):
        if detect_tuple is None:
            detect_tuple = self._last_detect_output
        if plane_name not in detect_tuple[0] or not detect_tuple[0][plane_name] or detect_tuple[0][plane_name]['ids'] is None or not detect_tuple[0][plane_name]['img_points']:
            return None, None
        objP, imgP = self._boards[plane_name].matchImagePoints(detect_tuple[0][plane_name]['img_points'], detect_tuple[0][plane_name]['ids'])
        if imgP is None or int(imgP.shape[0]/4)<self.planes[plane_name]['min_num_markers']:
            return None, None
        return objP, imgP

    def get_individual_marker_points(self, marker_id: int, detect_tuple=None):
        if detect_tuple is None:
            detect_tuple = self._last_detect_output
        if detect_tuple[1]['ids'] is None or not detect_tuple[1]['img_points'] or marker_id not in detect_tuple[1]['ids']:
            return None, None
        img_points = detect_tuple[1]['img_points'][detect_tuple[1]['ids'].flatten().tolist().index(marker_id)]
        return self._indiv_marker_points[marker_id], img_points

    def visualize(self, frame, detect_tuple=None, sub_pixel_fac=8, plane_marker_color=(0,255,0), recovered_plane_marker_color=(255,255,0), individual_marker_color=(255,0,255), unexpected_marker_color=(150,253,253), rejected_marker_color=None):
        if detect_tuple is None:
            detect_tuple = self._last_detect_output

        # for debug, can draw rejected markers on frame
        if rejected_marker_color is not None:
            cv2.aruco.drawDetectedMarkers(frame, detect_tuple[3], None, borderColor=rejected_marker_color)

        # draw detected markers on the frame
        if plane_marker_color is not None:
            for p in detect_tuple[0]:
                special_highlight = []
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
    # takes single planes, individual markers and detector settings, and consolidates them into a minimal
    # set of detectors with all planes/individual markers associated to one of these detectors
    # also handles information for each about when they should be detected (called intervals below)
    def __init__(self):
        self.working_set = WorkingSet()
        # planes to be detected
        self.planes                 : dict[str, PlaneSetup] = {}
        self.plane_proc_intervals   : dict[str, tuple[annotation.EventType, list[int]|list[list[int]]]|None] = {}
        self._plane_to_detector     : dict[str, int]        = {}
        # individual markers to be detected
        self.individual_markers                 : dict[marker.MarkerID, MarkerSetup] = {}
        self.individual_markers_proc_intervals  : dict[str, tuple[annotation.EventType, list[int]|list[list[int]]]|None] = {}
        self._individual_to_detector            : dict[marker.MarkerID, int] = {}

        # consolidated into set of detectors, and associated planes+individual markers for each
        self._detectors             : dict[int, Detector]                   = {}

        # colors for drawing (in BGR order)
        self._plane_marker_color            = (  0,255,  0)
        self._recovered_plane_marker_color  = (255,255,  0)
        self._individual_marker_color       = (255,  0,255)
        self._unexpected_marker_color       = (128,255,255)
        self._rejected_marker_color         = None # by default not drawn. If wanted, (0,0,255) is a good color

    def add_plane(self, plane: str, planes_setup: PlaneSetup, processing_intervals: tuple[annotation.EventType, list[int]|list[list[int]]]|None = None):
        if plane in self.planes:
            raise ValueError(f'Cannot register the plane "{plane}", it is already registered')
        self.planes[plane]                  = planes_setup
        self.plane_proc_intervals[plane]    = processing_intervals

    def add_individual_marker(self, mark: marker.MarkerID, marker_setup: MarkerSetup, processing_intervals: tuple[annotation.EventType, list[int]|list[list[int]]]|None = None):
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

    def consolidate_setup(self, allow_duplicated_markers: bool = False):
        # get list of all ArUco dicts and markers we're dealing with
        all_markers: set[marker.MarkerID] = set()
        err_msg = 'Markers are not unique across planes and individual markers'
        for p in self.planes:
            markers = self.planes[p]['plane'].get_marker_IDs()
            for ms in markers:
                if ms!='plane':
                    # N.B.: other markers should be registered by caller as individual markers
                    continue
                if (overlap := all_markers.intersection(markers[ms])):
                    t_err_msg = f'{err_msg} for plane "{p}", duplicated markers: {marker.format_duplicate_markers_msg({m.to_family() for m in overlap})}'
                    if allow_duplicated_markers:
                        print(f'Warning: {t_err_msg}')
                    else:
                        raise RuntimeError(t_err_msg)
                all_markers.update(markers[ms])
        for m in self.individual_markers:
            if m in all_markers:
                t_err_msg = f'{err_msg}: individual marker "{m}" is also used elsewhere'
                if allow_duplicated_markers:
                    print(f'Warning: {t_err_msg}')
                else:
                    raise RuntimeError(t_err_msg)
            all_markers.add(m)

        family_marker_ids: dict[int, set[int]] = {}
        for m in all_markers:
            family_marker_ids.setdefault(dict_id_to_family[m.aruco_dict_id], set()).add(m.m_id)

        registrations = [(self.planes[p]['plane'].aruco_dict_id, p, self.planes[p], True) for p in self.planes]
        registrations += [(m.aruco_dict_id, m, self.individual_markers[m], False) for m in self.individual_markers]
        groups = {}
        for dictionary_id, name, setup, is_plane in registrations:
            settings = resolve_settings(setup.get('aruco_settings') if is_plane else
                                        {'detector_params': setup.get('detector_params', {})})
            dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            key = (dict_id_to_family[dictionary_id], dictionary.maxCorrectionBits, freeze(settings))
            groups.setdefault(key, []).append((dictionary_id, name, setup, is_plane, settings))

        compatible_groups = []
        for entries in groups.values():
            bins = []
            for entry in entries:
                _, name, setup, is_plane, _ = entry
                for group in bins:
                    if is_plane or all(e[3] or e[1].m_id != name.m_id or
                                       (e[2].get('size'), e[2].get('detect_only', False)) ==
                                       (setup.get('size'), setup.get('detect_only', False)) for e in group):
                        group.append(entry)
                        break
                else:
                    bins.append([entry])
            compatible_groups.extend(bins)

        self._detectors.clear()
        self._plane_to_detector.clear()
        self._individual_to_detector.clear()
        self.working_set.clear()
        # Choose one raw dictionary per compatible detection group, even when refinement differs.
        raw_groups = {}
        for entries in groups.values():
            for d, _, _, _, settings in entries:
                dictionary = cv2.aruco.getPredefinedDictionary(d)
                key = (dict_id_to_family[d], dictionary.maxCorrectionBits, freeze(settings['detector_params']), settings['undistort'])
                raw_groups.setdefault(key, []).append(d)
        for detector_id, entries in enumerate(compatible_groups):
            d, _, _, _, settings = entries[0]
            dictionary = cv2.aruco.getPredefinedDictionary(d)
            raw_key = (dict_id_to_family[d], dictionary.maxCorrectionBits, freeze(settings['detector_params']), settings['undistort'])
            d = max(raw_groups[raw_key], key=get_dict_size)
            detector = Detector(d, settings)
            for _, name, setup, is_plane, _ in entries:
                if is_plane:
                    detector.add_plane(name, setup)
                    self._plane_to_detector[name] = detector_id
                else:
                    detector.add_individual_marker(name, setup)
                    self._individual_to_detector[name] = detector_id
            # Known IDs from other detectors in this family must not be drawn as unexpected.
            detector._all_markers.update(family_marker_ids[dict_id_to_family[d]])
            detector.create_detector()
            self._detectors[detector_id] = detector

    def register_with_estimator(self, estimator: pose.Estimator):
        if any(d.settings['undistort'] or d.settings['pose_consistency']['enabled'] for d in self._detectors.values()):
            if not estimator.cam_params.has_intrinsics():
                raise ValueError('Running ArUco detection on undistorted images and checking of pose consistency require camera calibration')
        # this handles registration of all planes and individual markers with the estimator
        # and makes sure our wrapper function for the aruco detector gets called which handles
        # aruco detection so that each detector is run only once on a frame
        for p in self.planes:
            estimator.add_plane(p,
                                lambda pn, fi, fr, finf, cp: self._detect_plane(pn, fi, fr, finf, cp),
                                self.plane_proc_intervals[p],
                                lambda pn, fi, fr, finf, _: self._visualize_plane(pn, fi, fr, finf))
        for m in self.individual_markers:
            estimator.add_individual_marker(m,
                                            lambda k, fi, fr, finf, cp: self._detect_individual_marker(k, fi, fr, finf, cp),
                                            self.individual_markers_proc_intervals[m],
                                            lambda k, fi, fr, finf, _: self._visualize_individual_marker(k, fi, fr, finf))

    def _detect_plane(self, plane_name: str, frame_idx: int, frame: np.ndarray, frame_info: dict, camera_parameters: ocv.CameraParams) -> pose.DetectionResult|None:
        if plane_name not in self._plane_to_detector:
            raise ValueError(f'The plane {plane_name} is not known')
        aruco_dict_id = self._plane_to_detector[plane_name]
        detect_tuple = self._get_detector_cache(aruco_dict_id, frame_idx, frame, frame_info, camera_parameters)
        if not detect_tuple[0] or plane_name not in detect_tuple[0] or not detect_tuple[0][plane_name]:
            return None
        obj, img = self._detectors[aruco_dict_id].get_matching_image_board_points(plane_name, detect_tuple)
        if obj is None or img is None:
            return None
        context = self.working_set.contexts[self._detectors[aruco_dict_id].settings['undistort']]
        return pose.DetectionResult(obj, img, context.camera_params, context.frame_info, context.undistorted)


    def _detect_individual_marker(self, mark: marker.MarkerID, frame_idx: int, frame: np.ndarray, frame_info: dict, camera_parameters: ocv.CameraParams) -> pose.DetectionResult|None:
        if mark not in self.individual_markers:
            raise ValueError(f'The individual marker {marker.marker_ID_to_str(mark)} is not known')
        detector_id = self._individual_to_detector[mark]
        detect_tuple = self._get_detector_cache(detector_id, frame_idx, frame, frame_info, camera_parameters)
        if not detect_tuple[1] or detect_tuple[1]['ids'] is None or mark.m_id not in detect_tuple[1]['ids']:
            return None
        detector = self._detectors[detector_id]
        obj, img = detector.get_individual_marker_points(mark.m_id, detect_tuple)
        if img is None:
            return None
        context = self.working_set.contexts[self._detectors[detector_id].settings['undistort']]
        return pose.DetectionResult(obj, img, context.camera_params, context.frame_info, context.undistorted)


    def _get_detector_cache(self, detector_id: int, frame_idx: int, frame: np.ndarray|None, frame_info: dict, camera_parameters: ocv.CameraParams|None):
        if frame is None:
            # Visualization only reads the current frame; it must not resurrect stale results.
            return self.working_set.detector_results.get(detector_id) if self.working_set.matches_frame(frame_idx) else None

        detector = self._detectors[detector_id]
        context = self.working_set.prepare(frame_idx, frame, frame_info, camera_parameters, detector.settings['undistort'])
        if detector_id not in self.working_set.detector_results:
            key = (detector.dictionary_id, freeze(detector.settings['detector_params']), context.undistorted)
            if key not in self.working_set.detections:
                self.working_set.detections[key] = detector._detect_markers(context.image)
            self.working_set.detector_results[detector_id] = detector.detect_markers(context.image, context.frame_info, context.camera_params, self.working_set.detections[key])
        return self.working_set.detector_results[detector_id]

    def _visualization_output(self, detector_id, detect_tuple):
        context = self.working_set.contexts[self._detectors[detector_id].settings['undistort']]
        if not context.undistorted:
            return detect_tuple
        planes, individual, unexpected, rejected = copy.deepcopy(detect_tuple)
        for points in [*planes.values(), individual, unexpected]:
            if points:
                points['img_points'] = context.original_corners(points['img_points'])
        return planes, individual, unexpected, context.original_corners(rejected)

    def _visualize_plane(self, plane_name: str, frame_idx: int, frame: np.ndarray, frame_info: dict):
        if plane_name not in self._plane_to_detector:
            raise ValueError(f'The plane {plane_name} is not known')
        aruco_dict_id = self._plane_to_detector[plane_name]
        if aruco_dict_id in self.working_set.visualized:
            # nothing to do, already drawn
            return
        detect_tuple = self._get_detector_cache(aruco_dict_id, frame_idx, None, frame_info, None)
        if detect_tuple is not None:
            frame = self._detectors[aruco_dict_id].visualize(frame, self._visualization_output(aruco_dict_id, detect_tuple), plane_marker_color=self._plane_marker_color, recovered_plane_marker_color=self._recovered_plane_marker_color, individual_marker_color=self._individual_marker_color, unexpected_marker_color=self._unexpected_marker_color, rejected_marker_color=self._rejected_marker_color)
            self.working_set.visualized.add(aruco_dict_id)

    def _visualize_individual_marker(self, mark: marker.MarkerID, frame_idx: int, frame: np.ndarray, frame_info: dict):
        if mark not in self.individual_markers:
            raise ValueError(f'The individual marker {marker.marker_ID_to_str(mark)} is not known')
        detector_id = self._individual_to_detector[mark]
        if detector_id in self.working_set.visualized:
            # nothing to do, already drawn
            return
        detect_tuple = self._get_detector_cache(detector_id, frame_idx, None, frame_info, None)
        if detect_tuple is not None:
            frame = self._detectors[detector_id].visualize(frame, self._visualization_output(detector_id, detect_tuple), plane_marker_color=self._plane_marker_color, recovered_plane_marker_color=self._recovered_plane_marker_color, individual_marker_color=self._individual_marker_color, unexpected_marker_color=self._unexpected_marker_color, rejected_marker_color=self._rejected_marker_color)
            self.working_set.visualized.add(detector_id)


def create_board(board_corner_points: list[np.ndarray], ids: list[int], ArUco_dict: cv2.aruco.Dictionary):
    board_corner_points = np.dstack(board_corner_points)        # list of 2D arrays -> 3D array
    board_corner_points = np.rollaxis(board_corner_points,-1)   # 4x2xN -> Nx4x2
    board_corner_points = np.pad(board_corner_points,((0,0),(0,0),(0,1)),'constant', constant_values=(0.,0.)) # Nx4x2 -> Nx4x3 (at Z=0 to all points)
    return cv2.aruco.Board(board_corner_points, ArUco_dict, np.array(ids))

def refine_detection(image: cv2.UMat, detected_corners, detected_ids, rejected_corners, det: cv2.aruco.ArucoDetector, board: cv2.aruco.Board, frame_info: dict, camera_parameters: ocv.CameraParams):
    # Corners must remain in the image's coordinates for OpenCV's pixel sampling.
    # Shift the principal point for a sensor ROI, rather than shifting the corners.
    camera_matrix = None
    distortion = None
    if camera_parameters.has_opencv_camera():
        camera_matrix = camera_parameters.camera_mtx.copy()
        camera_matrix[:2, 2] -= [frame_info.get('offset_x', 0), frame_info.get('offset_y', 0)]
        distortion = camera_parameters.distort_coeffs
    img_points, ids, rejected_img_points, _ = det.refineDetectedMarkers(
        image=image, board=board, detectedCorners=detected_corners, detectedIds=detected_ids,
        rejectedCorners=rejected_corners, cameraMatrix=camera_matrix, distCoeffs=distortion)
    if img_points and img_points[0].shape[0]==4:
        # there are versions out there where there is a bug in output shape of each set of corners, fix up
        img_points = [np.reshape(c,(1,4,2)) for c in img_points]
    if rejected_img_points and rejected_img_points[0].shape[0]==4:
        # same as for corners
        rejected_img_points = [np.reshape(c,(1,4,2)) for c in rejected_img_points]
    # determine which are new (recovered) markers
    recovered_ids = None
    if detected_ids is not None and ids is not None:
        recovered_ids = np.array(list(set(ids.flatten())-set(detected_ids.flatten()))).reshape((-1,1))

    return img_points, ids, rejected_img_points, recovered_ids

def filter_detections(img_points: list[np.ndarray], ids: np.ndarray, expected_ids: list[np.ndarray], keep_expected=True):
    if ids is None or not img_points:
        return img_points, ids
    if not keep_expected:
        expected_ids = set(ids.flatten())-set(expected_ids)
    if not expected_ids:
        # optimization. If output will definitely be empty, return directly
        return tuple(), None
    to_remove = np.where([x not in expected_ids for x in ids.flatten()])[0]
    ids = np.delete(ids, to_remove, axis=0)
    img_points = tuple(v.copy() for i,v in enumerate(img_points) if i not in to_remove)
    return img_points, ids


def has_duplicates(ids: np.ndarray) -> bool:
    """Return True if any ID repeats."""
    if ids is None or len(ids) == 0:
        return False
    flat = ids.flatten().astype(int)
    return len(flat) != len(np.unique(flat))

def group_indices_by_id(
    ids: np.ndarray
) -> dict[int, list[int]]:
    """
    Return {marker_id: [indices in the detections arrays]}.
    """
    id_to_indices: dict[int, list[int]] = {}
    if ids is None or len(ids) == 0:
        return id_to_indices
    for i, mid in enumerate(ids.flatten()):
        id_to_indices.setdefault(int(mid), []).append(i)
    return id_to_indices

def _build_board_objpoints_map(board: cv2.aruco.Board) -> dict[int, np.ndarray]:
    """
    Map marker_id -> (4,3) object points from the board.
    """
    return {int(mid): np.asarray(obj4x3, dtype=np.float32)
            for obj4x3, mid in zip(board.getObjPoints(), board.getIds().flatten())}

def _corners_4x2(c: np.ndarray) -> np.ndarray:
    """
    Normalize a detected corner array to shape (4,2) for error computation,
    while preserving the original array elsewhere.
    Accepts (4,1,2), (1,4,2), or (4,2).
    """
    c = np.asarray(c)
    if c.shape == (4, 2):
        return c.astype(np.float32)
    if c.shape == (4, 1, 2):
        return c.reshape(4, 2).astype(np.float32)
    if c.shape == (1, 4, 2):
        return c.reshape(4, 2).astype(np.float32)
    # Fallback: flatten last two dims to 2 cols if possible
    c2 = c.reshape(-1, 2)
    if c2.shape[0] == 4:
        return c2.astype(np.float32)
    raise ValueError(f"Unexpected corner shape: {c.shape}")

def _mean_corner_error_projected(
    observed_4x2: np.ndarray, projected_4x2: np.ndarray, test_rotations: bool = True
) -> float:
    """
    Mean L2 error between observed and projected corners. Optionally try 4 rotations of the observed
    corners to guard against corner order mismatches in inputs.
    """
    if not test_rotations:
        return np.linalg.norm(observed_4x2 - projected_4x2, axis=1).mean()

    best = float("inf")
    for rot in range(4):
        obs_rot = np.roll(observed_4x2, -rot, axis=0)
        err = np.linalg.norm(obs_rot - projected_4x2, axis=1).mean()
        if err < best:
            best = err
    return best

def _estimate_board_pose(
    board: cv2.aruco.Board,
    corners_list: list[np.ndarray],
    ids: np.ndarray,
    frame_info: dict,
    camera_params: ocv.CameraParams
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """INTERNAL: Estimate pose using all detections. Returns (ok, rvec, tvec)."""
    objP, imgP = board.matchImagePoints(corners_list, ids)
    if objP is None or len(objP) == 0:
        return False, None, None
    retval, rvec, tvec, _ = pose.estimate_pose(objP, imgP, frame_info, camera_params)
    if retval <= 0:
        return False, None, None
    return True, rvec, tvec

def filter_board_duplicates(
    board: cv2.aruco.Board,
    corners: list[np.ndarray],
    ids: np.ndarray,
    frame_info: dict,
    camera_params: ocv.CameraParams,
    *,
    min_markers: int = 3,
    max_combinations: int | None = 5000,
    test_corner_rotations: bool = True
) -> tuple[bool, list[np.ndarray], np.ndarray, list[int]]:
    """
    Exhaustive search over duplicate-ID choices. Single-ID detections are always included.
    For each duplicated ID, choose exactly one candidate detection. For each full combination,
    estimate board pose and compute mean reprojection error across all selected markers.
    Keep the combination with the smallest error.

    Returns:
        ok (bool),
        corners_consistent (list[np.ndarray]): preserved shapes per item (e.g., (4,1,2)),
        ids_consistent (np.ndarray): shape (N,1), dtype matches input ids.dtype,
        kept_indices (list[int]): indices into the original detections.

    Notes:
      - If the Cartesian product of duplicate choices exceeds `max_combinations`, returns False.
      - The scoring is mean per-corner L2 reprojection error (pixels) across all selected markers.
    """
    # --- Early outs ---
    if ids is None or len(ids) == 0:
        return False, [], np.empty((0, 1), dtype=np.int32), []

    # If all IDs are unique, keep all (no pose computation)
    if not has_duplicates(ids):
        kept_indices = list(range(len(ids)))
        corners_consistent = [corners[i] for i in kept_indices]
        ids_consistent = ids.reshape(-1, 1).copy()
        return True, corners_consistent, ids_consistent, []

    if len(corners) != len(ids):
        raise ValueError(f"corners (len={len(corners)}) and ids (len={len(ids)}) mismatch.")

    # Group detections by ID
    id2idx = group_indices_by_id(ids)
    singles: list[int] = []
    dup_groups: list[list[int]] = []
    for mid, idxs in id2idx.items():
        if len(idxs) == 1:
            singles.append(idxs[0])
        else:
            dup_groups.append(idxs)

    # Compute total combinations (product of lengths of duplicate groups)
    num_combinations = 1
    for g in dup_groups:
        num_combinations *= len(g)

    if max_combinations is not None and num_combinations > max_combinations:
        # Too many; caller assumed "few duplicates", so bail rather than blow up.
        return False, [], np.empty((0, 1), dtype=ids.dtype), []

    # Precompute object points per ID
    board_map = _build_board_objpoints_map(board)
    # Prepare an array form of ids for fast slicing
    ids_arr = ids.reshape(-1, 1)

    if 'offset_x' in frame_info and 'offset_y' in frame_info:
        # if we have a ROI, need to add the offset to the image points to get correct pose estimation
        ROI_offset = np.array([frame_info['offset_x'], frame_info['offset_y']])
    else:
        ROI_offset = np.array([0., 0.])

    # Iterate all choices, one index from each duplicate group
    best_err = float("inf")
    best_indices: list[int] = []
    for choice in itertools.product(*dup_groups):
        # Selected detections = singles + one per duplicated ID group
        selected = singles + list(choice)

        if len(selected) < min_markers:
            # Insufficient constraints to estimate pose robustly
            continue

        # Estimate pose for this combination
        sel_corners = [corners[i] for i in selected]
        sel_ids = ids_arr[selected]  # shape (K,1)

        retval, rvec, tvec = _estimate_board_pose(board, sel_corners, sel_ids, frame_info, camera_params)
        if retval <= 0:
            # Pose failed; skip this combination
            continue

        # Compute mean reprojection error for the selected markers
        # Project each selected marker's 3D corners with the pose and compare to observed
        total_err = 0.0
        total_corners = 0

        for i in selected:
            mid = int(ids_arr[i, 0])
            if mid not in board_map:
                # Board doesn't define this ID; skip (shouldn't happen if detections are on the same board)
                continue
            obj4x3 = board_map[mid]  # (4,3)
            proj4x2 = transforms.project_points(obj4x3, camera_params, rot_vec=rvec, trans_vec=tvec, ROI_offset=ROI_offset)

            obs4x2 = _corners_4x2(corners[i])
            err = _mean_corner_error_projected(obs4x2, proj4x2, test_rotations=test_corner_rotations)
            total_err += err * 4  # 4 corners
            total_corners += 4

        if total_corners == 0:
            # No valid error; skip
            continue

        mean_err = total_err / total_corners

        # Keep the lowest-error combination (tie-breaker: keep the one with more markers)
        if (mean_err < best_err) or (np.isclose(mean_err, best_err) and len(selected) > len(best_indices)):
            best_err = mean_err
            best_indices = selected

    if not best_indices:
        return False, [], np.empty((0, 1), dtype=ids.dtype), []

    # Prepare outputs (preserve corner shapes and return OpenCV-style ids)
    best_indices = sorted(best_indices)
    corners_consistent = [corners[i] for i in best_indices]
    ids_consistent = ids_arr[best_indices].reshape(-1, 1).astype(ids.dtype, copy=False)

    return True, corners_consistent, ids_consistent, list(set(range(len(ids)))-set(best_indices))


@dataclass
class ImageContext:
    image: np.ndarray
    frame_info: dict
    camera_params: ocv.CameraParams             # camera parameters for the image (which may be undistorted, in which case distortion will be zeroed)
    original_camera_params: ocv.CameraParams    # original camera parameters for the image, regardless of whether it was undistorted
    undistorted: bool = False

    def original_corners(self, corners):
        if not self.undistorted or corners is None:
            return corners
        offset = [self.frame_info.get('offset_x', 0), self.frame_info.get('offset_y', 0)]
        return tuple(transforms.distort_points(c.reshape(-1, 2), self.original_camera_params, offset).reshape(c.shape).astype(np.float32) for c in corners)


class WorkingSet:
    """One frame's images/detections, plus a cached calibration/ROI remap."""
    def __init__(self):
        self._frame_key = None
        self.contexts = {}                   # Raw and undistorted image contexts.
        self.detections = {}                 # Raw detections shared by compatible configurations.
        self.detector_results: dict[int, tuple] = {}  # Filtered/refined results per detector wrapper.
        self.visualized: set[int] = set()    # Detector wrappers already drawn on this frame.
        self._map_cache = None

    def _reset_frame(self):
        self._frame_key = None
        self.contexts.clear()
        self.detections.clear()
        self.detector_results.clear()
        self.visualized.clear()

    def clear(self):
        self._reset_frame()
        self._map_cache = None

    def matches_frame(self, frame_idx: int) -> bool:
        return self._frame_key is not None and self._frame_key[0] == frame_idx

    def prepare(self, frame_idx, image, frame_info, camera_params, undistort=False):
        colmap = camera_params.colmap_camera
        calibration = freeze((camera_params.resolution, camera_params.camera_mtx, camera_params.distort_coeffs,
                              None if colmap is None else (colmap.model_name, colmap.params)))
        frame_key = (frame_idx, id(image), freeze(frame_info), calibration)
        if frame_key != self._frame_key:
            self._reset_frame()
            self._frame_key = frame_key
        # always store the original image
        if False not in self.contexts:
            self.contexts[False] = ImageContext(image, dict(frame_info), camera_params, camera_params)
        # also store an undistorted image if requested and not already present
        if undistort and True not in self.contexts:
            if not camera_params.has_intrinsics():
                raise ValueError('ArUco image undistortion requires camera calibration')
            # make a copy of the camera parameters with distortion coefficients zeroed out
            effective = copy.copy(camera_params)
            if camera_params.has_opencv_camera():
                effective.distort_coeffs = np.zeros_like(camera_params.distort_coeffs)
            effective.colmap_camera = camera_params.colmap_camera_no_distortion
            # make undistortion mapping for the image
            map_key = (image.shape[:2], frame_info.get('offset_x', 0), frame_info.get('offset_y', 0), calibration)
            if self._map_cache is None or self._map_cache[0] != map_key:
                height, width = image.shape[:2]
                yy, xx = np.indices((height, width), dtype=np.float32)
                pixels = np.column_stack((xx.ravel(), yy.ravel()))
                source = transforms.distort_points(pixels, camera_params, map_key[1:3])
                maps = source.reshape(height, width, 2).astype(np.float32)
                self._map_cache = (map_key, cv2.convertMaps(maps, None, cv2.CV_16SC2))
            # undistort the image
            working_image = cv2.remap(image, *self._map_cache[1], interpolation=cv2.INTER_LINEAR)
            self.contexts[True] = ImageContext(working_image, dict(frame_info), effective, camera_params, True)
        return self.contexts[undistort]