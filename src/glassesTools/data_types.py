# NB: using pose information requires a calibrated scene camera
import enum

from . import json, utils as _utils


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
                   "For each time point, take angular offset between the left and " \
                   "right gaze positions and the fixation target and average them " \
                   "to compute angular measures in degrees. Requires " \
                   f"'{ler_name}' and '{rer_name}' to be enabled."