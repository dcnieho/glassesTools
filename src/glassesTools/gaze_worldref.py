import numpy as np
import math
import cv2
import pathlib
import typing
import enum

from . import data_files, drawing, gaze_headref, ocv, plane, transforms, utils

class Type(utils.AutoName):
    Scene_Video_Position    = enum.auto()
    World_3D_Point          = enum.auto()
    Left_Eye_Gaze_Vector    = enum.auto()
    Right_Eye_Gaze_Vector   = enum.auto()
    Average_Gaze_Vector     = enum.auto()
utils.register_type(utils.CustomTypeEntry(Type,'__enum.gaze_worldref.Type__', utils.enum_val_2_str, lambda x: getattr(Type, x.split('.')[-1])))

class Gaze:
    # description of tsv file used for storage
    _columns_compressed = {'timestamp': 1, 'timestamp_VOR': 1, 'timestamp_ref': 1, 'frame_idx':1, 'frame_idx_VOR':1, 'frame_idx_ref':1,
                           'gazePosCam_vidPos_ray':3,'gazePosCam_vidPos_homography':3,'gazePosCamWorld':3,'gazeOriCamLeft':3,'gazePosCamLeft':3,'gazeOriCamRight':3,'gazePosCamRight':3,
                           'gazePosPlane2D_vidPos_ray':2,'gazePosPlane2D_vidPos_homography':2,'gazePosPlane2DWorld':2,'gazePosPlane2DLeft':2,'gazePosPlane2DRight':2}
    _non_float          = {'frame_idx': int, 'frame_idx_VOR': int, 'frame_idx_ref': int}
    _columns_optional   = ['timestamp_VOR', 'frame_idx_VOR', 'timestamp_ref', 'frame_idx_ref']

    def __init__(self,
                 timestamp                          : float,
                 frame_idx                          : int,

                 timestamp_ori                      : float      = None,
                 frame_idx_ori                      : int        = None,
                 timestamp_VOR                      : float      = None,
                 frame_idx_VOR                      : int        = None,
                 timestamp_ref                      : float      = None,
                 frame_idx_ref                      : int        = None,

                 gazePosCam_vidPos_ray              : np.ndarray = None,
                 gazePosCam_vidPos_homography       : np.ndarray = None,
                 gazePosCamWorld                    : np.ndarray = None,
                 gazeOriCamLeft                     : np.ndarray = None,
                 gazePosCamLeft                     : np.ndarray = None,
                 gazeOriCamRight                    : np.ndarray = None,
                 gazePosCamRight                    : np.ndarray = None,

                 gazePosPlane2D_vidPos_ray          : np.ndarray = None,
                 gazePosPlane2D_vidPos_homography   : np.ndarray = None,
                 gazePosPlane2DWorld                : np.ndarray = None,
                 gazePosPlane2DLeft                 : np.ndarray = None,
                 gazePosPlane2DRight                : np.ndarray = None):

        self.timestamp                          = timestamp
        self.frame_idx                          = frame_idx

        # optional timestamps and frame_idxs
        self.timestamp_ori                      = timestamp_ori                     # timestamp field can be set to any of these three. Keep here a copy of the original timestamp
        self.frame_idx_ori                      = frame_idx_ori
        self.timestamp_VOR                      = timestamp_VOR
        self.frame_idx_VOR                      = frame_idx_VOR
        self.timestamp_ref                      = timestamp_ref
        self.frame_idx_ref                      = frame_idx_ref                     # frameidx _in reference video_ not in this eye tracker's video (unless this is the reference)

        # in camera space (3D coordinates)
        self.gazePosCam_vidPos_ray              = gazePosCam_vidPos_ray             # video gaze position on plane (camera ray intersected with plane)
        self.gazePosCam_vidPos_homography       = gazePosCam_vidPos_homography      # gaze2DHomography in camera space
        self.gazePosCamWorld                    = gazePosCamWorld                   # 3D gaze point on plane (world-space gaze point, turned into direction ray and intersected with plane)
        self.gazeOriCamLeft                     = gazeOriCamLeft
        self.gazePosCamLeft                     = gazePosCamLeft                    # 3D gaze point on plane ( left eye gaze vector intersected with plane)
        self.gazeOriCamRight                    = gazeOriCamRight
        self.gazePosCamRight                    = gazePosCamRight                   # 3D gaze point on plane (right eye gaze vector intersected with plane)

        # in plane space (2D coordinates)
        self.gazePosPlane2D_vidPos_ray          = gazePosPlane2D_vidPos_ray         # Video gaze point mapped to plane by turning into direction ray and intersecting with plane
        self.gazePosPlane2D_vidPos_homography   = gazePosPlane2D_vidPos_homography  # Video gaze point directly mapped to plane through homography transformation
        self.gazePosPlane2DWorld                = gazePosPlane2DWorld               # gazePosCamWorld in plane space
        self.gazePosPlane2DLeft                 = gazePosPlane2DLeft                # gazePosCamLeft in plane space
        self.gazePosPlane2DRight                = gazePosPlane2DRight               # gazePosCamRight in plane space

    def get_gaze_point(self, gaze_type: Type, reference_frame='plane'):
        return_3D = reference_frame in ('world', 'camera')    # in all other cases return gaze on plane
        match gaze_type:
            case Type.Scene_Video_Position:
                gaze_point = self.gazePosCam_vidPos_ray if return_3D else self.gazePosPlane2D_vidPos_ray
                if gaze_point is None:
                    # fall back to homography
                    gaze_point = self.gazePosCam_vidPos_homography if return_3D else self.gazePosPlane2D_vidPos_homography
            case Type.World_3D_Point:
                gaze_point = self.gazePosCamWorld if return_3D else self.gazePosPlane2DWorld
            case Type.Left_Eye_Gaze_Vector:
                gaze_point = self.gazePosCamLeft if return_3D else self.gazePosPlane2DLeft
            case Type.Right_Eye_Gaze_Vector:
                gaze_point = self.gazePosCamRight if return_3D else self.gazePosPlane2DRight
            case Type.Average_Gaze_Vector:
                gaze_point = None
                if return_3D and (self.gazePosCamLeft is not None) and (self.gazePosCamRight is not None):
                    gaze_point = (self.gazePosCamLeft, self.gazePosCamRight)
                elif not return_3D and (self.gazePosPlane2DLeft is not None) and (self.gazePosPlane2DRight is not None):
                    gaze_point = (self.gazePosPlane2DLeft, self.gazePosPlane2DRight)
                if gaze_point is not None:
                    gaze_point = (gaze_point[0]+gaze_point[1])/2
        return gaze_point

    def draw_on_world_video(self, img, camera_params: ocv.CameraParams, sub_pixel_fac=1, pose: plane.Pose=None,
                            clr_vidPos=(255,255,0), clr_world_pos=(255,0,255), clr_left=(0,0,255), clr_right=(255,0,0), clr_average=(255,0,255)):
        # project to camera, display
        def _project(pos):
            return cv2.projectPoints(pos.reshape(1,3), np.zeros((1,3)),np.zeros((1,3)), camera_params.camera_mtx,camera_params.distort_coeffs)[0].flatten()
        def _draw(img,cam_pos,sz,clr):
            if not math.isnan(cam_pos[0]):
                drawing.openCVCircle(img, cam_pos, sz, clr, -1, sub_pixel_fac)
        def project_and_draw(img,pos,sz,clr):
            _draw(img,_project(pos),sz,clr)

        if not camera_params.has_intrinsics():
            if pose is not None and pose.homography_successful():
                # gaze position on plane by homography
                _draw(img,pose.plane_to_cam_homography(self.gazePosPlane2D_vidPos_homography,camera_params),3,clr_vidPos)
            # rest requires camera cal, so return
            return

        # gaze ray
        if self.gazePosCam_vidPos_ray is not None and clr_vidPos is not None:
            project_and_draw(img, self.gazePosCam_vidPos_ray, 3, clr_vidPos)
        # binocular gaze point
        if self.gazePosCamWorld is not None and clr_world_pos is not None:
            project_and_draw(img, self.gazePosCamWorld, 3, clr_world_pos)
        # left eye
        if self.gazePosCamLeft is not None and clr_left is not None:
            project_and_draw(img, self.gazePosCamLeft, 3, clr_left)
        # right eye
        if self.gazePosCamRight is not None and clr_right is not None:
            project_and_draw(img, self.gazePosCamRight, 3, clr_right)
        # average
        if clr_average is not None and (pointCam:=self.get_gaze_point(Type.Average_Gaze_Vector, 'world')) is not None:
            project_and_draw(img, pointCam, 6, clr_average)

    def draw_on_plane(self, img, reference, sub_pixel_fac=1):
        # binocular gaze point
        if self.gazePosPlane2DWorld is not None:
            reference.draw(img, *self.gazePosPlane2DWorld, sub_pixel_fac, (0,255,255), 3)
        # left eye
        if self.gazePosPlane2DLeft is not None:
            reference.draw(img, *self.gazePosPlane2DLeft, sub_pixel_fac, (0,0,255), 3)
        # right eye
        if self.gazePosPlane2DRight is not None:
            reference.draw(img, *self.gazePosPlane2DRight, sub_pixel_fac, (255,0,0), 3)
        # average
        if (average:=self.get_gaze_point(Type.Average_Gaze_Vector)) is not None:
            if not math.isnan(average[0]):
                reference.draw(img, *average, sub_pixel_fac, (255,0,255))
        # video gaze position
        if self.gazePosPlane2D_vidPos_homography is not None:
            reference.draw(img, *self.gazePosPlane2D_vidPos_homography, sub_pixel_fac, (0,255,0), 5)
        if self.gazePosPlane2D_vidPos_ray is not None:
            reference.draw(img, *self.gazePosPlane2D_vidPos_ray, sub_pixel_fac, (255,255,0), 3)

def read_dict_from_file(file_name:str|pathlib.Path, episodes:list[list[int]]=None, ts_column_suffixes: list[str] = None) -> dict[int,list[Gaze]]:
    return data_files.read_file(file_name,
                                Gaze, False, False, True, True,
                                episodes=episodes, ts_fridx_field_suffixes=ts_column_suffixes)[0]

def write_dict_to_file(gazes: list[Gaze] | dict[int,list[Gaze]], file_name:str|pathlib.Path, skip_missing=False):
    data_files.write_array_to_file(gazes, file_name,
                                   Gaze._columns_compressed,
                                   Gaze._columns_optional,
                                   skip_all_nan=skip_missing)

@typing.overload
def from_head(poses:           plane.Pose , gazes_head:               gaze_headref.Gaze  , camera_params: ocv.CameraParams) ->               Gaze  : ...
@typing.overload
def from_head(poses: dict[int, plane.Pose], gazes_head: dict[int,list[gaze_headref.Gaze]], camera_params: ocv.CameraParams) -> dict[int,list[Gaze]]: ...

def from_head(poses: plane.Pose|dict[int, plane.Pose], gazes: gaze_headref.Gaze|dict[int,list[gaze_headref.Gaze]], camera_params: ocv.CameraParams) -> Gaze|dict[int,list[Gaze]]:
    if not isinstance(poses, dict):
        return _from_head_impl(poses, gazes, camera_params)

    world_gazes = {}
    for frame_idx in poses:
        if frame_idx in gazes:
            world_gazes[frame_idx] = []
            for gaze in gazes[frame_idx]:
                gaze_world = _from_head_impl(poses[frame_idx], gaze, camera_params)
                world_gazes[frame_idx].append(gaze_world)
    return world_gazes

def _from_head_impl(pose: plane.Pose, gaze: gaze_headref.Gaze, camera_params: ocv.CameraParams) -> Gaze:
    gaze_world = Gaze(gaze.timestamp, gaze.frame_idx, gaze.timestamp_ori, gaze.frame_idx_ori, gaze.timestamp_VOR, gaze.frame_idx_VOR, gaze.timestamp_ref, gaze.frame_idx_ref)
    if pose.pose_successful():
        # get transform from ET data's coordinate frame to camera's coordinate frame
        camera_rotation = camera_params.rotation_vec
        camera_position = camera_params.position
        if camera_rotation is None:
            camera_rotation = np.zeros((3,1))
        RCam  = cv2.Rodrigues(camera_rotation)[0]
        if camera_position is None:
            camera_position = np.zeros((3,1))
        RtCam = np.hstack((RCam, camera_position))

        # project gaze on video to reference plane using camera pose
        gaze_world.gazePosPlane2D_vidPos_ray, gaze_world.gazePosCam_vidPos_ray = \
            pose.cam_to_plane_pose(gaze.gaze_pos_vid, camera_params)

        # project world-space gaze point (often binocular gaze point) to plane
        if gaze.gaze_pos_3d is not None:
            # transform 3D gaze point from eye tracker space to camera space
            g3D = np.matmul(RtCam,np.array(np.append(gaze.gaze_pos_3d, 1)).reshape(4,1))

            # find intersection with plane (NB: pose is in camera reference frame)
            gaze_world.gazePosCamWorld = pose.vector_intersect(g3D)    # default vec origin (0,0,0) is fine because we work from camera's view point

            # above intersection is in camera space, turn into plane space to get position on plane
            (x,y,z) = pose.cam_frame_to_world(gaze_world.gazePosCamWorld)   # z should be very close to zero
            gaze_world.gazePosPlane2DWorld = np.asarray([x, y])

    # unproject 2D gaze point on video to point on plane (should yield values very close to
    # the above method of intersecting video gaze point ray with plane, and usually also very
    # close to binocular gaze point (though for at least one tracker the latter is not the case;
    # the AdHawk has an optional parallax correction using a vergence signal))
    if pose.homography_successful():
        gaze_world.gazePosPlane2D_vidPos_homography = pose.cam_to_plane_homography(gaze.gaze_pos_vid, camera_params)

        # get this point in camera space
        if pose.pose_successful():
            gaze_world.gazePosCam_vidPos_homography = pose.world_frame_to_cam(np.append(gaze_world.gazePosPlane2D_vidPos_homography, 0))

    # project gaze vectors to plane
    if not pose.pose_successful():
        # nothing to do anymore
        return gaze_world

    gaze_vecs    = [gaze.gaze_dir_l, gaze.gaze_dir_r]
    gaze_origins = [gaze.gaze_ori_l, gaze.gaze_ori_r]
    attrs        = [['gazeOriCamLeft','gazePosCamLeft','gazePosPlane2DLeft'],['gazeOriCamRight','gazePosCamRight','gazePosPlane2DRight']]
    for gVec,gOri,attr in zip(gaze_vecs,gaze_origins,attrs):
        if gVec is None or gOri is None:
            continue
        # get gaze vector and point on vector (origin, e.g. pupil center) ->
        # transform from ET data coordinate frame into camera coordinate frame
        gVec    = np.matmul(RCam ,          gVec    )
        gOri    = np.matmul(RtCam,np.append(gOri,1.))
        setattr(gaze_world,attr[0],gOri)

        # intersect with plane -> yield point on plane in camera reference frame
        gPlane = pose.vector_intersect(gVec, gOri)
        setattr(gaze_world,attr[1],gPlane)

        # transform intersection with plane from camera space to plane space
        (x,y,z)  = pose.cam_frame_to_world(gPlane)  # z should be very close to zero
        setattr(gaze_world,attr[2],np.asarray([x, y]))

    return gaze_world

def distance_from_plane(gaze: Gaze, plane: plane.Plane):
    if gaze.gazePosPlane2D_vidPos_ray is not None and not math.isnan(gaze.gazePosPlane2D_vidPos_ray[0]):
        gp = gaze.gazePosPlane2D_vidPos_ray
    elif gaze.gazePosPlane2D_vidPos_homography is not None and not math.isnan(gaze.gazePosPlane2D_vidPos_homography[0]):
        gp = gaze.gazePosPlane2D_vidPos_homography
    else:
        return math.nan
    return transforms.dist_from_bbox(*gp, plane.bbox)
