import enum

import numpy as np
from matplotlib import colors
import pathlib
import typing
import math
import cv2
import pandas as pd
from collections import defaultdict

from .. import aruco, drawing as _drawing, json, marker as _marker, plane as _plane, transforms as _transforms, utils as _utils

from . import config
from . import default_poster
from . import dynamic

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
json.register_type(json.TypeEntry(DataQualityType, 'glassesValidator.DataQualityType', _utils.enum_val_2_str, lambda x: _utils.enum_str_2_val(x, DataQualityType)))

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


class Plane(_plane.Plane):
    def __init__(self, config_dir: str|pathlib.Path|None, validation_config: dict[str,typing.Any]=None, is_dynamic=False, **kwarg):
        # NB: if config_dir is None, the default config will be used

        if config_dir is not None:
            config_dir = pathlib.Path(config_dir)

        # get validation config, if needed
        if validation_config is None:
            validation_config = config.get_validation_setup(config_dir)
        self.config = validation_config

        # get marker width
        if self.config['mode'] == 'deg':
            self.cell_size_mm = 2.*math.tan(math.radians(.5))*self.config['distance']*10
        else:
            self.cell_size_mm = 10 # 1cm
        markerSize = self.cell_size_mm*self.config['markerSide']

        # get board size
        plane_size = _plane.Coordinate(self.config['gridCols']*self.cell_size_mm, self.config['gridRows']*self.cell_size_mm)

        # get targets first, so that they can be drawn on the reference image
        self.targets: dict[int,_marker.Marker]                  = {}
        self.dynamic_markers: dict[int, tuple[int,int]]         = {}        # {marker ID: (target ID, marker_N column in target file)} (keep latter around for good error reporting)
        self._dynamic_markers_cache: dict[int, _marker.MarkerID]= None      # different format, for efficient return from get_marker_IDs()
        origin = self._get_targets(config_dir, self.config, is_dynamic)

        # call base class
        markers = config.get_markers(config_dir, self.config['markerPosFile'])
        if 'ref_image_store_path' not in kwarg:
            kwarg['ref_image_store_path'] = None
        super(Plane, self).__init__(markers, markerSize, plane_size, self.config['arucoDictionary'], self.config['markerBorderBits'], self.cell_size_mm, "mm", ref_image_size=self.config['referencePosterSize'], min_num_markers=self.config['minNumMarkers'], **kwarg)

        # set center
        self.set_origin(origin)

    def set_origin(self, origin: _plane.Coordinate):
        # set origin of plane. Origin location is on current (not original) plane
        # so set_origin([5., 0.]) three times in a row shifts the origin rightward by 15 units
        for i in self.targets:
            self.targets[i].shift(-np.array(origin))
        super(Plane, self).set_origin(origin)

    def _get_targets(self, config_dir, validationSetup, is_dynamic) -> _plane.Coordinate:
        """ poster space: (0,0) is origin (might be center target), (-,-) bottom left """

        # read in target positions
        targets = config.get_targets(config_dir, validationSetup['targetPosFile'])
        if targets is not None:
            targets['center'] = list(targets[['x','y']].values)
            targets['center'] *= self.cell_size_mm
            if is_dynamic:
                # split of columns indicating markers that signal appearance of a target
                markers = pd.concat([targets.pop(c) for c in targets.columns if c.startswith('marker_')], axis=1)
            targets = targets.drop([x for x in targets.columns if x not in ('center','color')], axis=1)
            self.targets = {idx:_marker.Marker(idx,**kwargs) for idx,kwargs in zip(targets.index.values,targets.to_dict(orient='records'))}
            origin = _plane.Coordinate(*targets.loc[validationSetup['centerTarget']].center.copy())  # NB: need origin in scaled space
            # load with dynamic markers, if any
            if is_dynamic:
                marker_columns = {c:int(c.removeprefix('marker_')) for c in markers}
                def _store_markers(r: pd.Series):
                    for c in marker_columns:
                        self.dynamic_markers[int(r[c])] = (int(r.name), marker_columns[c])
                markers.apply(_store_markers, axis=1)
        else:
            self.targets.clear()
            self.dynamic_markers.clear()
            origin = _plane.Coordinate(0.,0.)
        self._dynamic_markers_cache = None
        return origin

    def _store_reference_image(self, path: pathlib.Path, width: int) -> np.ndarray:
        # first call superclass method to generate image without targets
        img = super(Plane, self)._store_reference_image(path, width)
        height = img.shape[0]

        # add targets
        subPixelFac = 8   # for sub-pixel positioning
        for key in self.targets:
            # check we're on the plane
            if np.any(self.targets[key].center[0]<0) or np.any(self.targets[key].center[0]>self.plane_size.x) or \
               np.any(self.targets[key].center[1]<0) or np.any(self.targets[key].center[1]>self.plane_size.y):
                center  = ", ".join(map(lambda x: f"{x:.4f}",self.targets[key].center))
                plane_corners = [", ".join(map(lambda x: f"{x:.4f}",c)) for c in (self.bbox[:2],self.bbox[2:])]
                raise ValueError(f'Target {key} positioned at ({center}) is outside the defined\nplane which ranges from ({plane_corners[0]}) to ({plane_corners[1]}). Ensure all\nsizes and positions are in the same unit (e.g. mm) and check the target position csv file and plane size.')

            # 1. determine position on image
            circlePos = _transforms.to_image_pos(*self.targets[key].center, self.bbox, [width,height])

            # 2. draw
            clr = tuple([int(i*255) for i in (colors.to_rgb(self.targets[key].color)[::-1] if self.targets[key].color else (0.,0.,1.))])  # need BGR color ordering
            _drawing.openCVCircle(img, circlePos, 15, clr, -1, subPixelFac)

        if path:
            cv2.imwrite(path, img)

        return img

    def get_marker_IDs(self) -> dict[str|int,list[_marker.MarkerID]]:
        if self._dynamic_markers_cache is None:
            self._dynamic_markers_cache = defaultdict(list)
            # {marker ID: (target ID, marker_N column in target file)} -> {marker_N column in target file: [(marker_id, aruco_dict)]}
            for m in self.dynamic_markers:
                self._dynamic_markers_cache[self.dynamic_markers[m][1]].append(_marker.MarkerID(m, self.aruco_dict_id))
        return super(Plane, self).get_marker_IDs() | self._dynamic_markers_cache

    def is_dynamic(self):
        return not not self.dynamic_markers

    def get_dynamic_marker_setup(self):
        return aruco.MarkerSetup(aruco_detector_params = {
                                    'markerBorderBits': self.marker_border_bits
                                },
                                detect_only = True
                             )