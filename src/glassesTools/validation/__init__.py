import enum

from . import config
from .. import utils as _utils

# NB: using pose information requires a calibrated scene camera
class DataQualityType(enum.Enum):
    viewpos_vidpos_homography   = enum.auto()   # use homography to map gaze from video to poster, and viewing distance defined in config (combined with the assumptions that the viewing position (eye) is located directly in front of the poster's center and that the poster is oriented perpendicularly to the line of sight) to compute angular measures
    pose_vidpos_homography      = enum.auto()   # use homography to map gaze from video to poster, and pose information w.r.t. poster to compute angular measures
    pose_vidpos_ray             = enum.auto()   # use camera calibration to map gaze position on scene video to cyclopean gaze vector, and pose information w.r.t. poster to compute angular measures
    pose_world_eye              = enum.auto()   # use provided gaze position in world (often a binocular gaze point), and pose information w.r.t. poster to compute angular measures
    pose_left_eye               = enum.auto()   # use provided left eye gaze vector, and pose information w.r.t. poster to compute angular measures
    pose_right_eye              = enum.auto()   # use provided right eye gaze vector, and pose information w.r.t. poster to compute angular measures
    pose_left_right_avg         = enum.auto()   # report average of left (pose_left_eye) and right (pose_right_eye) eye angular values

    # so this gets serialized in a user-friendly way by pandas..
    def __str__(self):
        return self.name
_utils.register_type(_utils.CustomTypeEntry(DataQualityType, 'glassesValidator.DataQualityType', _utils.enum_val_2_str, lambda x: _utils.enum_str_2_val(x, DataQualityType)))

def get_DataQualityType_explanation(dq: DataQualityType):
    ler_name =  "Left eye ray + pose"
    rer_name = "Right eye ray + pose"
    match dq:
        case DataQualityType.viewpos_vidpos_homography:
            return "Homography + view distance", \
                   "Use homography to map gaze position from the scene video to " \
                   "the validation poster, and use an assumed viewing distance (see " \
                   "the project's configuration) to compute data quality measures " \
                   "in degrees with respect to the scene camera. In this mode, it is "\
                   "assumed that the eye is located exactly in front of the center of "\
                   "the poster and that the poster is oriented perpendicularly to the "\
                   "line of sight from this assumed viewing position."
        case DataQualityType.pose_vidpos_homography:
            return "Homography + pose", \
                   "Use homography to map gaze position from the scene video to " \
                   "the validation poster, and use the determined pose of the scene " \
                   "camera (requires a calibrated camera) to compute data quality " \
                   "measures in degrees with respect to the scene camera."
        case DataQualityType.pose_vidpos_ray:
            return "Video ray + pose", \
                   "Use camera calibration to turn gaze position from the scene " \
                   "video into a direction vector, and determine gaze position on " \
                   "the validation poster by intersecting this vector with the " \
                   "poster. Then, use the determined pose of the scene camera " \
                   "(requires a calibrated camera) to compute data quality " \
                   "measures in degrees with respect to the scene camera."
        case DataQualityType.pose_world_eye:
            return "World gaze position + pose", \
                   "Use the gaze position in the world provided by the eye tracker " \
                   "(often a binocular gaze point) to determine gaze position on the " \
                   "validation poster by turning it into a direction vector with respect " \
                   "to the scene camera and intersecting this vector with the poster. " \
                   "Then, use the determined pose of the scene camera " \
                   "(requires a calibrated camera) to compute data quality " \
                   "measures in degrees with respect to the scene camera."
        case DataQualityType.pose_left_eye:
            return ler_name, \
                   "Use the gaze direction vector for the left eye provided by " \
                   "the eye tracker to determine gaze position on the validation " \
                   "poster by intersecting this vector with the poster. " \
                   "Then, use the determined pose of the scene camera " \
                   "(requires a camera calibration) to compute data quality " \
                   "measures in degrees with respect to the left eye."
        case DataQualityType.pose_right_eye:
            return rer_name, \
                   "Use the gaze direction vector for the right eye provided by " \
                   "the eye tracker to determine gaze position on the validation " \
                   "poster by intersecting this vector with the poster. " \
                   "Then, use the determined pose of the scene camera " \
                   "(requires a camera calibration) to compute data quality " \
                   "measures in degrees with respect to the right eye."
        case DataQualityType.pose_left_right_avg:
            return "Average eye rays + pose", \
                   "For each time point, take angular offset between the left and " \
                   "right gaze positions and the fixation target and average them " \
                   "to compute data quality measures in degrees. Requires " \
                   f"'{ler_name}' and '{rer_name}' to be enabled."