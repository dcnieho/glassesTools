import numpy as np
import math
import cv2
import pathlib
import typing

from . import data_files, drawing, gaze_headref, ocv, plane

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
        # 3D gaze is in world space, w.r.t. scene camera
        # 2D gaze is on the poster
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

        # in poster space (2D coordinates)
        self.gazePosPlane2D_vidPos_ray          = gazePosPlane2D_vidPos_ray         # Video gaze point mapped to poster by turning into direction ray and intersecting with poster
        self.gazePosPlane2D_vidPos_homography   = gazePosPlane2D_vidPos_homography  # Video gaze point directly mapped to poster through homography transformation
        self.gazePosPlane2DWorld                = gazePosPlane2DWorld               # gazePosCamWorld in poster space
        self.gazePosPlane2DLeft                 = gazePosPlane2DLeft                # gazePosCamLeft in poster space
        self.gazePosPlane2DRight                = gazePosPlane2DRight               # gazePosCamRight in poster space

    def draw_on_world_video(self, img, camera_params: ocv.CameraParams, sub_pixel_fac=1):
        # project to camera, display
        def project_and_draw(img,pos,sz,clr):
            pPointCam = cv2.projectPoints(pos.reshape(1,3),np.zeros((1,3)),np.zeros((1,3)),camera_params.camera_mtx, camera_params.distort_coeffs)[0][0][0]
            if not math.isnan(pPointCam[0]):
                drawing.openCVCircle(img, pPointCam, sz, clr, -1, sub_pixel_fac)

        # gaze ray
        if self.gazePosCam_vidPos_ray is not None:
            project_and_draw(img, self.gazePosCam_vidPos_ray, 3, (255,255,0))
        # binocular gaze point
        if self.gazePosCamWorld is not None:
            project_and_draw(img, self.gazePosCamWorld, 3, (255,0,255))
        # left eye
        if self.gazePosCamLeft is not None:
            project_and_draw(img, self.gazePosCamLeft, 3, (0,0,255))
        # right eye
        if self.gazePosCamRight is not None:
            project_and_draw(img, self.gazePosCamRight, 3, (255,0,0))
        # average
        if (self.gazePosCamLeft is not None) and (self.gazePosCamRight is not None):
            pointCam  = np.array([(x+y)/2 for x,y in zip(self.gazePosCamLeft,self.gazePosCamRight)])
            project_and_draw(img, pointCam, 6, (255,0,255))

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
        if (self.gazePosPlane2DLeft is not None) and (self.gazePosPlane2DRight is not None):
            average = np.array([(x1+x2)/2 for x1,x2 in zip(self.gazePosPlane2DLeft,self.gazePosPlane2DRight)])
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

        # project gaze on video to reference poster using camera pose
        gaze_world.gazePosPlane2D_vidPos_ray, gaze_world.gazePosCam_vidPos_ray = \
            pose.cam_to_plane_pose(gaze.gaze_pos_vid, camera_params)

        # project world-space gaze point (often binocular gaze point) to plane
        if gaze.gaze_pos_3d is not None:
            # transform 3D gaze point from eye tracker space to camera space
            g3D = np.matmul(RtCam,np.array(np.append(gaze.gaze_pos_3d, 1)).reshape(4,1))

            # find intersection with poster (NB: pose is in camera reference frame)
            gaze_world.gazePosCamWorld = pose.vector_intersect(g3D)    # default vec origin (0,0,0) is fine because we work from camera's view point

            # above intersection is in camera space, turn into poster space to get position on poster
            (x,y,z) = pose.cam_frame_to_world(gaze_world.gazePosCamWorld)   # z should be very close to zero
            gaze_world.gazePosPlane2DWorld = np.asarray([x, y])

    # unproject 2D gaze point on video to point on poster (should yield values very close to
    # the above method of intersecting video gaze point ray with poster, and usually also very
    # close to binocular gaze point (though for at least one tracker the latter is not the case;
    # the AdHawk has an optional parallax correction using a vergence signal))
    if pose.homography_successful():
        gaze_world.gazePosPlane2D_vidPos_homography = pose.cam_to_plane_homography(gaze.gaze_pos_vid, camera_params)

        # get this point in camera space
        if pose.pose_successful():
            gaze_world.gazePosCam_vidPos_homography = pose.world_frame_to_cam(np.append(gaze_world.gazePosPlane2D_vidPos_homography, 0))

    # project gaze vectors to reference poster (and draw on video)
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

        # intersect with poster -> yield point on poster in camera reference frame
        gPoster = pose.vector_intersect(gVec, gOri)
        setattr(gaze_world,attr[1],gPoster)

        # transform intersection with poster from camera space to poster space
        (x,y,z)  = pose.cam_frame_to_world(gPoster)  # z should be very close to zero
        setattr(gaze_world,attr[2],np.asarray([x, y]))

    return gaze_world