# NB: using pose information requires a calibrated scene camera
import enum
import typing
import numpy as np
from ETDQualitizer import vector_to_Fick

from . import json, pose, transforms, utils as _utils, gaze_worldref


# type of data to use for computing angular measures
class DataType(_utils.AutoName):
    viewpos_vidpos_homography   = enum.auto()   # use homography to map gaze from video to plane, and viewing distance defined in config (combined with the assumptions that the viewing position (eye) is located directly in front of the plane's center and that the plane is oriented perpendicularly to the line of sight) to compute angular measures
    pose_vidpos_homography      = enum.auto()   # use homography to map gaze from video to plane, and pose information w.r.t. plane to compute angular measures
    pose_vidpos_ray             = enum.auto()   # use camera calibration to map gaze position on scene video to cyclopean gaze vector, and pose information w.r.t. plane to compute angular measures
    pose_world_eye              = enum.auto()   # use provided gaze position in world (often a binocular gaze point), and pose information w.r.t. plane to compute angular measures
    pose_left_eye               = enum.auto()   # use provided left eye gaze vector, and pose information w.r.t. plane to compute angular measures
    pose_right_eye              = enum.auto()   # use provided right eye gaze vector, and pose information w.r.t. plane to compute angular measures
    pose_left_right_avg         = enum.auto()   # report average of left (pose_left_eye) and right (pose_right_eye) eye angular values

    # so this gets serialized in a user-friendly way by pandas..
    def __str__(self):
        return self.name
def data_type_val_to_enum_val(x: int|str) -> DataType:
    return _utils.str_int_2_enum_val(x, DataType, {1:'viewpos_vidpos_homography', 2:'pose_vidpos_homography', 3: 'pose_vidpos_ray', 4: 'pose_world_eye', 5: 'pose_left_eye', 6: 'pose_right_eye', 7: 'pose_left_right_avg'})
json.register_type(json.TypeEntry(DataType, '__enum.DataType__', _utils.enum_val_2_str, data_type_val_to_enum_val, compatible_reg_names=['glassesValidator.DataQualityType']))


def get_explanation(dq: DataType):
    ler_name =  "Left eye ray + pose"
    rer_name = "Right eye ray + pose"
    match dq:
        case DataType.viewpos_vidpos_homography:
            return "Homography + view distance", \
                   "Use homography to map gaze position from the scene video to " \
                   "the validation plane, and use an assumed viewing distance (see " \
                   "the project's configuration) to compute angular measures " \
                   "in degrees with respect to the scene camera. In this mode, it is "\
                   "assumed that the eye is located exactly in front of the center of "\
                   "the plane and that the plane is oriented perpendicularly to the "\
                   "line of sight from this assumed viewing position."
        case DataType.pose_vidpos_homography:
            return "Homography + pose", \
                   "Use homography to map gaze position from the scene video to " \
                   "the validation plane, and use the determined pose of the scene " \
                   "camera (requires a calibrated camera) to compute angular " \
                   "measures in degrees with respect to the scene camera."
        case DataType.pose_vidpos_ray:
            return "Video ray + pose", \
                   "Use camera calibration to turn gaze position from the scene " \
                   "video into a direction vector, and determine gaze position on " \
                   "the validation plane by intersecting this vector with the " \
                   "plane. Then, use the determined pose of the scene camera " \
                   "(requires a calibrated camera) to compute angular " \
                   "measures in degrees with respect to the scene camera."
        case DataType.pose_world_eye:
            return "World gaze position + pose", \
                   "Use the gaze position in the world provided by the eye tracker " \
                   "(often a binocular gaze point) to determine gaze position on the " \
                   "validation plane by turning it into a direction vector with respect " \
                   "to the scene camera and intersecting this vector with the plane. " \
                   "Then, use the determined pose of the scene camera " \
                   "(requires a calibrated camera) to compute angular " \
                   "measures in degrees with respect to the scene camera."
        case DataType.pose_left_eye:
            return ler_name, \
                   "Use the gaze direction vector for the left eye provided by " \
                   "the eye tracker to determine gaze position on the validation " \
                   "plane by intersecting this vector with the plane. " \
                   "Then, use the determined pose of the scene camera " \
                   "(requires a camera calibration) to compute angular " \
                   "measures in degrees with respect to the left eye."
        case DataType.pose_right_eye:
            return rer_name, \
                   "Use the gaze direction vector for the right eye provided by " \
                   "the eye tracker to determine gaze position on the validation " \
                   "plane by intersecting this vector with the plane. " \
                   "Then, use the determined pose of the scene camera " \
                   "(requires a camera calibration) to compute angular " \
                   "measures in degrees with respect to the right eye."
        case DataType.pose_left_right_avg:
            return "Average eye rays + pose", \
                   "For each time point, take the spherical midpoint of the left " \
                   "and right gaze directions before computing target-relative " \
                   "angular measures in degrees. Requires " \
                   f"'{ler_name}' and '{rer_name}' to be enabled."


def get_world_gaze_fields_for_data_type(angle_type: DataType) -> list[str|None]:
    # field 1: origin of gaze vector (None if scene camera)
    # field 2: 3D gaze point in camera space (None if not available)
    # field 3: 2D gaze point on plane in plane space
    match angle_type:
        case DataType.viewpos_vidpos_homography:
            # from camera perspective, using homography
            # viewpos_vidpos_homography: using assumed viewing distance
            fields = [None, None, 'gazePosPlane2D_vidPos_homography']
        case DataType.pose_vidpos_homography:
            # from camera perspective, using homography
            # pose_vidpos_homography   : using pose info
            fields = [None, 'gazePosCam_vidPos_homography', 'gazePosPlane2D_vidPos_homography']
        case DataType.pose_vidpos_ray:
            # from camera perspective, using 3D gaze point ray
            fields = [None, 'gazePosCam_vidPos_ray', 'gazePosPlane2D_vidPos_ray']
        case DataType.pose_world_eye:
            # using 3D world gaze position, with respect to eye tracker reference frame's origin
            fields = [None, 'gazePosCamWorld', 'gazePosPlane2DWorld']
        case DataType.pose_left_eye:
            fields = ['gazeOriCamLeft', 'gazePosCamLeft', 'gazePosPlane2DLeft']
        case DataType.pose_right_eye:
            fields = ['gazeOriCamRight', 'gazePosCamRight', 'gazePosPlane2DRight']
        case _:
            raise NotImplementedError(f'Logic for gaze angle type {angle_type} not implemented. Contact developer.')
    return fields

def get_available_data_types(plane_gazes: dict[int, list[gaze_worldref.Gaze]]) -> list[DataType]:
    dq_have: list[DataType] = []
    for dq in DataType:
        if dq == DataType.pose_left_right_avg:
            continue   # special case handled below (needs both left and right eye data)
        fields = get_world_gaze_fields_for_data_type(dq)
        have_data = np.vstack(tuple([(a:=getattr(s,f)) is not None and not np.any(np.isnan(a)) for v in plane_gazes.values() for s in v] for f in fields if f is not None))
        if np.any(np.all(have_data,axis=0)):
            dq_have.append(dq)

    if (DataType.pose_left_eye in dq_have) and (DataType.pose_right_eye in dq_have):
        dq_have.append(DataType.pose_left_right_avg)
    return dq_have

def select_data_types_to_use(dq_types: typing.Iterable[DataType]|DataType|str|None, dq_have: list[DataType], allow_dq_fallback: bool = True) -> list[DataType]:
    if dq_types is not None:
        if isinstance(dq_types,DataType) or isinstance(dq_types, str):
            dq_types = [dq_types]
        else:
            # ensure list
            dq_types = list(dq_types)

    if dq_types:
        # do some checks on user input
        for i,dq in reversed(list(enumerate(dq_types))):
            if not isinstance(dq, DataType):
                if isinstance(dq, str):
                    if hasattr(DataType, dq):
                        dq = dq_types[i] = getattr(DataType, dq)
                    else:
                        raise ValueError(f"The string '{dq}' is not a known data type. Known types: {[e.name for e in DataType]}")
                else:
                    raise ValueError(f"The variable 'dq' should be a string with one of the following values: {[e.name for e in DataType]}")
            if not dq in dq_have:
                if allow_dq_fallback:
                    del dq_types[i]
                else:
                    raise RuntimeError(f'Data type {dq} could not be used as its not available for this recording. Available data types: {[e.name for e in dq_have]}')

        if DataType.pose_left_right_avg in dq_types:
            if (not DataType.pose_left_eye in dq_have) or (not DataType.pose_right_eye in dq_have):
                if allow_dq_fallback:
                    dq_types.remove(DataType.pose_left_right_avg)
                else:
                    raise RuntimeError(f'Cannot use the data type {DataType.pose_left_right_avg} because it requires having data types {DataType.pose_left_eye} and {DataType.pose_right_eye} available, but one or both are not available. Available data types: {[e.name for e in dq_have]}')

    if not dq_types:
        if DataType.pose_vidpos_ray in dq_have:
            # highest priority is DataType.pose_vidpos_ray
            dq_types.append(DataType.pose_vidpos_ray)
        elif DataType.pose_vidpos_homography in dq_have:
            # else at least try to use pose (shouldn't occur, if we have pose we have a calibrated camera, which means we should have the above)
            dq_types.append(DataType.pose_vidpos_homography)
        else:
            # else we're down to falling back on an assumed viewing distance
            if not DataType.viewpos_vidpos_homography in dq_have:
                raise RuntimeError(f'Even data type {DataType.viewpos_vidpos_homography} could not be used, bare minimum failed for some weird reason. Contact developer.')
            dq_types.append(DataType.viewpos_vidpos_homography)

    return dq_types


def calculate_target_relative_gaze_vectors(plane_gazes: dict[int, list[gaze_worldref.Gaze]], poses: dict[int, pose.Pose] | dict[int, list[pose.Pose]], points: dict[int, np.ndarray], d_types: list[DataType], points_for_homography: dict[int, np.ndarray]|None=None, viewing_distance: float|None = None) -> tuple[list[int], np.ndarray, dict[int, dict[DataType, np.ndarray]]]:
    requested_types = list(dict.fromkeys(d_types))
    selected_types = [d_type for d_type in requested_types if d_type != DataType.pose_left_right_avg]
    if DataType.pose_left_right_avg in requested_types:
        selected_types.extend(d_type for d_type in (DataType.pose_left_eye, DataType.pose_right_eye) if d_type not in selected_types)

    selected_data: dict[DataType, tuple[list[int], list[float], np.ndarray, np.ndarray]] = {}
    for d_type in selected_types:
        selected_data[d_type] = collect_gaze_data(plane_gazes, d_type, viewing_distance)
    if selected_data:
        first_data = next(iter(selected_data.values()))
        frame_idxs = first_data[0]
        timestamps = np.asarray(first_data[1], dtype=float)
    else:
        frame_idxs = []
        timestamps = np.empty((0,), dtype=float)

    out: dict[int, dict[DataType, np.ndarray]] = {}
    for target_id, target_point in points.items():
        target_vectors: dict[int,dict[int,np.ndarray]] = {}
        computed_vectors = {}
        for d_type, (sample_frames, sample_timestamps, origins, gaze_points) in selected_data.items():
            vectors = np.full((len(sample_frames), 3), np.nan)
            for sample_idx, frame_idx in enumerate(sample_frames):
                if d_type == DataType.viewpos_vidpos_homography:
                    if points_for_homography is None:
                        raise ValueError('points_for_homography must be provided for homography gaze data.')
                    gaze_vector   = gaze_points[sample_idx]
                    target_vector = points_for_homography[target_id]
                else:
                    this_pose = pose.get_sample_pose(poses, frame_idx, timestamp=float(sample_timestamps[sample_idx]))
                    if this_pose is None:
                        continue
                    if frame_idx not in target_vectors:
                        target_vectors[frame_idx] = {}
                    if sample_idx not in target_vectors[frame_idx]:
                        target_vectors[frame_idx][sample_idx] = this_pose.world_frame_to_cam(target_point)
                    gaze_vector   = gaze_points[sample_idx]               - origins[sample_idx]
                    target_vector = target_vectors[frame_idx][sample_idx] - origins[sample_idx]

                vectors[sample_idx] = transforms.vector_to_target_frame(gaze_vector, target_vector)
            computed_vectors[d_type] = vectors

        out[target_id] = {}
        for d_type in requested_types:
            if d_type == DataType.pose_left_right_avg:
                out[target_id][d_type] = average_gaze_vectors(computed_vectors[DataType.pose_left_eye], computed_vectors[DataType.pose_right_eye])
            else:
                out[target_id][d_type] = computed_vectors[d_type]

    return frame_idxs, timestamps, out

def target_relative_vectors_to_fick(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=float)
    if vectors.shape[-1] != 3:
        raise ValueError('Target-relative gaze vectors should have a final dimension of 3.')
    return vector_to_Fick(vectors[..., 0], vectors[..., 1], vectors[..., 2])

def average_gaze_vectors(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_norm = np.linalg.norm(left, axis=-1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=-1, keepdims=True)
    valid = (np.all(np.isfinite(left), axis=-1, keepdims=True) & np.all(np.isfinite(right), axis=-1, keepdims=True) & (left_norm > 0) & (right_norm > 0))
    left_unit  = np.divide( left,  left_norm, out=np.zeros_like( left), where= left_norm > 0)
    right_unit = np.divide(right, right_norm, out=np.zeros_like(right), where=right_norm > 0)
    midpoint = left_unit + right_unit
    midpoint_norm = np.linalg.norm(midpoint, axis=-1, keepdims=True)
    valid &= midpoint_norm > 1e-12
    return np.divide(midpoint, midpoint_norm, out=np.full_like(midpoint, np.nan), where=valid)

def calculate_gaze_angles_to_point(plane_gazes: dict[int, list[gaze_worldref.Gaze]], poses: dict[int, pose.Pose] | dict[int, list[pose.Pose]], points: dict[int, np.ndarray], d_types: list[DataType], points_for_homography: dict[int, np.ndarray]|None=None, viewing_distance: float|None = None) -> tuple[list[int], np.ndarray, dict[int, dict[DataType, np.ndarray]]]:
    frame_idxs, timestamps, vectors = calculate_target_relative_gaze_vectors(plane_gazes, poses, points, d_types, points_for_homography, viewing_distance)
    out: dict[int, dict[DataType, np.ndarray]] = {}
    for target_id, target_vectors in vectors.items():
        out[target_id] = {}
        for d_type, gaze_vectors in target_vectors.items():
            azi, ele = target_relative_vectors_to_fick(gaze_vectors)
            total = np.degrees(np.arctan2(np.hypot(gaze_vectors[:, 0], gaze_vectors[:, 1]), gaze_vectors[:, 2]))
            out[target_id][d_type] = np.column_stack((total, azi, ele))
    return frame_idxs, timestamps, out

def collect_gaze_data(plane_gazes: dict[int, list[gaze_worldref.Gaze]], d_type: DataType, viewing_distance: float|None = None) -> tuple[list[int], list[float], np.ndarray, np.ndarray]:
    if d_type == DataType.viewpos_vidpos_homography and viewing_distance is None:
        raise ValueError(f'When using data type {DataType.viewpos_vidpos_homography}, a viewing distance (in mm) should be provided.')

    frame_idxs      = [k           for k,v in plane_gazes.items() for _ in v]
    ts              = [s.timestamp for  v in plane_gazes.values() for s in v]

    fields          = get_world_gaze_fields_for_data_type(d_type)
    if fields[0] is None:
        ori     = np.zeros((len(ts),3))
    else:
        ori     = np.vstack([getattr(s,fields[0]) for v in plane_gazes.values() for s in v])
    if fields[1] is None:
        if not d_type==DataType.viewpos_vidpos_homography:
            raise NotImplementedError(f'This field should be set, is a special case not implemented? Contact developer')
        assert fields[2] is not None, 'Field 2 should never be None. Contact developer'
        gazePlane   = np.vstack([getattr(s,fields[2]) for v in plane_gazes.values() for s in v])
        gaze        = np.hstack((gazePlane[:,0:2], np.full((gazePlane.shape[0],1),viewing_distance)))
    else:
        gaze    = np.vstack([getattr(s,fields[1]) for v in plane_gazes.values() for s in v])

    return frame_idxs, ts, ori, gaze
