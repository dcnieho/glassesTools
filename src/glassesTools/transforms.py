import numpy as np
import cv2

from . import marker

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
            img_points.extend( [x.flatten() for x in dc] )
            obj_points.extend( known[detected_IDs[i]].corners )

    if len(img_points) < 4:
        return None, False

    # compute Homography
    return estimate_homography(obj_points, img_points)

def estimate_homography(obj_points, img_points):
    img_points = np.float32(img_points)
    obj_points = np.float32(obj_points)
    h, _ = cv2.findHomography(img_points, obj_points)
    return h, True

def apply_homography(H, x, y):
    if np.isnan(x) or np.isnan(y):
        return np.full((2,), np.nan)

    src = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    dst = cv2.perspectiveTransform(src,H)
    return dst.flatten()


def distort_point(x, y, camera_matrix, dist_coeff):
    if np.isnan(x) or np.isnan(y):
        return np.full((2,), np.nan)

    # unproject, ignoring distortion as this is an undistored point
    points_2d = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    points_2d = cv2.undistortPoints(points_2d, camera_matrix, np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]]))
    points_3d = cv2.convertPointsToHomogeneous(points_2d)
    points_3d.shape = -1, 3

    # reproject, applying distortion
    points_2d, _ = cv2.projectPoints(points_3d, np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), camera_matrix, dist_coeff)
    return points_2d.flatten()

def undistort_point(x, y, camera_matrix, dist_coeff):
    if np.isnan(x) or np.isnan(y):
        return np.full((2,), np.nan)

    points_2d = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    points_2d = cv2.undistortPoints(points_2d, camera_matrix, dist_coeff, P=camera_matrix) # P=cameraMatrix to reproject to camera
    return points_2d.flatten()

def unproject_point(x, y, camera_matrix, dist_coeff):
    if np.isnan(x) or np.isnan(y):
        return np.full((3,), np.nan)

    points_2d = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    points_2d = cv2.undistortPoints(points_2d, camera_matrix, dist_coeff)
    points_3d = cv2.convertPointsToHomogeneous(points_2d)
    points_3d.shape = -1, 3
    return points_3d.flatten()


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