import numpy as np
import cv2
import typing

from . import marker, ocv

M = typing.TypeVar("M", bound=int)
N = typing.TypeVar("N", bound=int)

def to_norm_pos(x,y,bbox):
    # transforms input (x,y) which is on a plane in world units
    # (e.g. mm on an aruco poster) to a normalized position
    # in an image of the plane, given the image's bounding box in
    # world units
    # for input (0,0) is bottom left, for output (0,0) is top left
    # bbox is [left, top, right, bottom]

    extents = [bbox[2]-bbox[0], bbox[1]-bbox[3]]
    pos     = [(x-bbox[0])/extents[0], (bbox[1]-y)/extents[1]]    # bbox[1]-y instead of y-bbox[3] to flip y
    return pos

def to_image_pos(x,y,bbox,img_size,margin=[0,0]):
    # transforms input (x,y) which is on a plane in world units
    # (e.g. mm on an aruco poster) to a pixel position in the
    # image, given the image's bounding box in world units
    # imSize should be active image area in pixels, excluding margin

    # fractional position between bounding box edges, (0,0) in bottom left
    pos = to_norm_pos(x,y, bbox)
    # turn into int, add margin
    pos = [p*s+m for p,s,m in zip(pos,img_size,margin)]
    return pos

def in_bbox(x,y,bbox,margin=0.):
    pos = to_norm_pos(x,y,bbox)
    return (pos[0]>=-margin and pos[0]<=1+margin) and (pos[1]>=-margin and pos[1]<=1+margin)

def dist_from_bbox(x,y,bbox):
    pos = to_norm_pos(x,y,bbox)
    if (pos[0]>=0 and pos[0]<=1) and (pos[1]>=0 and pos[1]<=1):
        return 0.   # inside bbox
    # compute max distance from edge of bbox
    dx = pos[0] if pos[0]<0. else pos[0]-1
    dy = pos[1] if pos[1]<0. else pos[1]-1
    return abs(max(dx,dy))


def estimate_homography_known_marker(known: list[marker.Marker], detected_corners, detected_IDs):
    # collect matching corners in image and in world
    img_points = []
    obj_points = []
    detected_IDs = detected_IDs.flatten()
    if len(detected_IDs) != len(detected_corners):
        raise ValueError('unexpected number of IDs (%d) given number of corner arrays (%d)' % (len(detected_IDs),len(detected_corners)))
    for i in range(0, len(detected_IDs)):
        if detected_IDs[i] in known:
            dc = detected_corners[i]
            if dc.shape[0]==1 and dc.shape[1]==4:
                dc = np.reshape(dc,(4,1,2))
            img_points.extend([x.flatten() for x in dc])
            obj_points.extend(known[detected_IDs[i]].corners)

    if len(img_points) < 4:
        return None

    # compute Homography
    return estimate_homography(obj_points, img_points)

def estimate_homography(obj_points, img_points):
    img_points = np.float32(img_points)
    obj_points = np.float32(obj_points)
    h, _ = cv2.findHomography(img_points, obj_points)
    return h

def apply_homography(points, H):
    if np.any(np.isnan(points)):
        return np.full_like(points, np.nan)

    return cv2.perspectiveTransform(points.astype('float').reshape((-1,1,2)),H).reshape((-1,2))


def distort_points(points_cam: np.ndarray[tuple[M, typing.Literal[2]], np.dtype[np.float64]], cam_params: ocv.CameraParams) -> np.ndarray[tuple[M, typing.Literal[2]], np.dtype[np.float64]]:
    if np.any(np.isnan(points_cam)):
        return np.full_like(points_cam, np.nan)

    if cam_params.has_colmap_camera():
        # unproject, ignoring distortion as this is an undistorted point
        points_w = cam_params.colmap_camera_no_distortion.cam_from_img(points_cam.reshape((-1,2)))
        # reproject, applying distortion
        return cam_params.colmap_camera.img_from_cam(points_w)
    elif cam_params.has_opencv_camera():
        # unproject, ignoring distortion as this is an undistorted point
        points_w = cv2.undistortPoints(points_cam.astype('float'), cam_params.camera_mtx, np.zeros((1, 5)))
        # reproject, applying distortion
        return cv2.projectPoints(cv2.convertPointsToHomogeneous(points_w), np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), cam_params.camera_mtx, cam_params.distort_coeffs)[0].reshape((-1,2))
    else:
        return np.full_like(points_cam, np.nan)

def undistort_points(points_cam: np.ndarray[tuple[M, typing.Literal[2]], np.dtype[np.float64]], cam_params: ocv.CameraParams) -> np.ndarray[tuple[M, typing.Literal[2]], np.dtype[np.float64]]:
    if np.any(np.isnan(points_cam)):
        return np.full_like(points_cam, np.nan)

    if cam_params.has_colmap_camera():
        # unproject, removing distortion
        points_w = cam_params.colmap_camera.cam_from_img(points_cam.reshape((-1,2)))
        # reproject, without applying distortion
        return cam_params.colmap_camera_no_distortion.img_from_cam(points_w)
    elif cam_params.has_opencv_camera():
        return cv2.undistortPoints(points_cam.astype('float'), cam_params.camera_mtx, cam_params.distort_coeffs, P=cam_params.camera_mtx).reshape((-1,2)) # P=cameraMatrix to reproject to camera
    else:
        return np.full_like(points_cam, np.nan)

def unproject_points(points_cam: np.ndarray[tuple[M, typing.Literal[2]], np.dtype[np.float64]], cam_params: ocv.CameraParams) -> np.ndarray[tuple[M, typing.Literal[3]], np.dtype[np.float64]]:
    if np.any(np.isnan(points_cam)):
        return np.full((points_cam.shape[0],3), np.nan)

    if cam_params.has_colmap_camera():
        return cv2.convertPointsToHomogeneous(cam_params.colmap_camera.cam_from_img(points_cam.reshape((-1,2)))).reshape((-1,3))
    elif cam_params.has_opencv_camera():
        points_w = cv2.undistortPoints(points_cam.reshape((-1,2)).astype('float'), cam_params.camera_mtx, cam_params.distort_coeffs).reshape((-1,2))
        return cv2.convertPointsToHomogeneous(points_w).reshape((-1,3))
    else:
        return np.full((points_cam.shape[0],3), np.nan)

def project_points(points_world: np.ndarray[tuple[M, typing.Literal[3]], np.dtype[np.float64]], cam_params: ocv.CameraParams, ignore_distortion=False, rot_vec: np.ndarray[tuple[typing.Literal[3]], np.dtype[np.float64]]=None, trans_vec: np.ndarray[tuple[typing.Literal[3]], np.dtype[np.float64]]=None) -> np.ndarray[tuple[M, typing.Literal[2]], np.dtype[np.float64]]:
    if np.any(np.isnan(points_world)):
        return np.full((points_world.shape[0],2), np.nan)

    if cam_params.has_colmap_camera():
        if rot_vec is not None and trans_vec is not None:
            RMat = cv2.Rodrigues(rot_vec)[0]
            RtMat = np.hstack((RMat, trans_vec.reshape(3,1)))
            points_world = np.matmul(RtMat,cv2.convertPointsToHomogeneous(points_world.reshape((-1,3))).reshape((-1,4)).T).T
        if ignore_distortion:
            return cam_params.colmap_camera_no_distortion.img_from_cam(points_world.reshape((-1,3)))
        else:
            return cam_params.colmap_camera.img_from_cam(points_world.reshape((-1,3))).reshape((-1,2))
    elif cam_params.has_opencv_camera():
        return cv2.projectPoints(points_world.astype('float'),
                                 np.zeros((1, 1, 3)) if rot_vec is None else rot_vec,
                                 np.zeros((1, 1, 3)) if trans_vec is None else trans_vec,
                                 cam_params.camera_mtx,
                                 np.zeros((1, 5)) if ignore_distortion else cam_params.distort_coeffs)[0].reshape((-1,2))
    else:
        return np.full((points_world.shape[0],2), np.nan)


def intersect_plane_ray(plane_normal, plane_point, ray_direction, ray_point, epsilon=1e-6):
    # from https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python

    ndotu = plane_normal.dot(ray_direction)
    if abs(ndotu) < epsilon:
        # raise RuntimeError("no intersection or line is within plane")
        return np.full((3,), np.nan)

    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    return w + si * ray_direction + plane_point


def angle_between(v1, v2):
    return (180.0 / np.pi) * np.arctan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2))