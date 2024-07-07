import numpy as np
import math
import cv2

from . import gaze_headref, gaze_worldref, marker, ocv, plane

def toNormPos(x,y,bbox):
    # transforms input (x,y) which is on a plane in world units
    # (e.g. mm on an aruco poster) to a normalized position
    # in an image of the plane, given the image's bounding box in
    # world units
    # for input (0,0) is bottom left, for output (0,0) is top left
    # bbox is [left, top, right, bottom]

    extents = [bbox[2]-bbox[0], bbox[1]-bbox[3]]
    pos     = [(x-bbox[0])/extents[0], (bbox[1]-y)/extents[1]]    # bbox[1]-y instead of y-bbox[3] to flip y
    return pos

def toImagePos(x,y,bbox,imSize,margin=[0,0]):
    # transforms input (x,y) which is on a plane in world units
    # (e.g. mm on an aruco poster) to a pixel position in the
    # image, given the image's bounding box in world units
    # imSize should be active image area in pixels, excluding margin

    # fractional position between bounding box edges, (0,0) in bottom left
    pos = toNormPos(x,y, bbox)
    # turn into int, add margin
    pos = [p*s+m for p,s,m in zip(pos,imSize,margin)]
    return pos


def estimateHomographyKnownMarker(known: list[marker.Marker], detectedCorners, detectedIDs):
    # collect matching corners in image and in world
    imgP = []
    objP = []
    detectedIDs = detectedIDs.flatten()
    if len(detectedIDs) != len(detectedCorners):
        raise ValueError('unexpected number of IDs (%d) given number of corner arrays (%d)' % (len(detectedIDs),len(detectedCorners)))
    for i in range(0, len(detectedIDs)):
        if detectedIDs[i] in known:
            dc = detectedCorners[i]
            if dc.shape[0]==1 and dc.shape[1]==4:
                dc = np.reshape(dc,(4,1,2))
            imgP.extend( [x.flatten() for x in dc] )
            objP.extend( known[detectedIDs[i]].corners )

    if len(imgP) < 4:
        return None, False

    # compute Homography
    return estimateHomography(objP, imgP)

def estimateHomography(objP, imgP):
    imgP = np.float32(imgP)
    objP = np.float32(objP)
    h, _ = cv2.findHomography(imgP, objP)
    return h, True

def applyHomography(H, x, y):
    if math.isnan(x) or math.isnan(y):
        return np.full((2,), np.nan)

    src = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    dst = cv2.perspectiveTransform(src,H)
    return dst.flatten()


def distortPoint(x, y, cameraMatrix, distCoeff):
    if math.isnan(x) or math.isnan(y):
        return np.full((2,), np.nan)

    # unproject, ignoring distortion as this is an undistored point
    points_2d = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    points_2d = cv2.undistortPoints(points_2d, cameraMatrix, np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]]))
    points_3d = cv2.convertPointsToHomogeneous(points_2d)
    points_3d.shape = -1, 3

    # reproject, applying distortion
    points_2d, _ = cv2.projectPoints(points_3d, np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), cameraMatrix, distCoeff)
    return points_2d.flatten()

def undistortPoint(x, y, cameraMatrix, distCoeff):
    if math.isnan(x) or math.isnan(y):
        return np.full((2,), np.nan)

    points_2d = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    points_2d = cv2.undistortPoints(points_2d, cameraMatrix, distCoeff, P=cameraMatrix) # P=cameraMatrix to reproject to camera
    return points_2d.flatten()

def unprojectPoint(x, y, cameraMatrix, distCoeff):
    if math.isnan(x) or math.isnan(y):
        return np.full((3,), np.nan)

    points_2d = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    points_2d = cv2.undistortPoints(points_2d, cameraMatrix, distCoeff)
    points_3d = cv2.convertPointsToHomogeneous(points_2d)
    points_3d.shape = -1, 3
    return points_3d.flatten()


def intersect_plane_ray(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    # from https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python

    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        # raise RuntimeError("no intersection or line is within plane")
        return np.full((3,), np.nan)

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    return w + si * rayDirection + planePoint


def angle_between(v1, v2):
    return (180.0 / math.pi) * math.atan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2))


def gazeToPlane(gaze: gaze_headref.Gaze, pose: plane.Pose, cameraParams: ocv.CameraParams) -> gaze_worldref.Gaze:
    gazeWorld     = gaze_worldref.Gaze(gaze.timestamp, gaze.frame_idx, gaze.timestamp_ori, gaze.frame_idx_ori, gaze.timestamp_VOR, gaze.frame_idx_VOR, gaze.timestamp_ref, gaze.frame_idx_ref)
    if pose.pose_N_markers>0:
        # get transform from ET data's coordinate frame to camera's coordinate frame
        cameraRotation = cameraParams.rotation_vec
        cameraPosition = cameraParams.position
        if cameraRotation is None:
            cameraRotation = np.zeros((3,1))
        RCam  = cv2.Rodrigues(cameraRotation)[0]
        if cameraPosition is None:
            cameraPosition = np.zeros((3,1))
        RtCam = np.hstack((RCam, cameraPosition))

        # project gaze on video to reference poster using camera pose
        gazeWorld.gazePosPlane2D_vidPos_ray, gazeWorld.gazePosCam_vidPos_ray = \
            pose.camToPlanePose(gaze.gaze_pos_vid, cameraParams)

        # project world-space gaze point (often binocular gaze point) to plane
        if gaze.gaze_pos_3d is not None:
            # transform 3D gaze point from eye tracker space to camera space
            g3D = np.matmul(RtCam,np.array(np.append(gaze.gaze_pos_3d, 1)).reshape(4,1))

            # find intersection with poster (NB: pose is in camera reference frame)
            gazeWorld.gazePosCamWorld = pose.vectorIntersect(g3D)    # default vec origin (0,0,0) is fine because we work from camera's view point

            # above intersection is in camera space, turn into poster space to get position on poster
            (x,y,z) = pose.camToWorld(gazeWorld.gazePosCamWorld)   # z should be very close to zero
            gazeWorld.gazePosPlane2DWorld = np.asarray([x, y])

    # unproject 2D gaze point on video to point on poster (should yield values very close to
    # the above method of intersecting video gaze point ray with poster, and usually also very
    # close to binocular gaze point (though for at least one tracker the latter is not the case;
    # the AdHawk has an optional parallax correction using a vergence signal))
    if pose.homography_N_markers>0:
        gazeWorld.gazePosPlane2D_vidPos_homography = pose.camToPlaneHomography(gaze.gaze_pos_vid, cameraParams)

        # get this point in camera space
        if pose.pose_N_markers>0:
            gazeWorld.gazePosCam_vidPos_homography = pose.worldToCam(np.append(gazeWorld.gazePosPlane2D_vidPos_homography, 0))

    # project gaze vectors to reference poster (and draw on video)
    if not pose.pose_N_markers>0:
        # nothing to do anymore
        return gazeWorld

    gazeVecs    = [gaze.gaze_dir_l, gaze.gaze_dir_r]
    gazeOrigins = [gaze.gaze_ori_l, gaze.gaze_ori_r]
    attrs       = [['gazeOriCamLeft','gazePosCamLeft','gazePosPlane2DLeft'],['gazeOriCamRight','gazePosCamRight','gazePosPlane2DRight']]
    for gVec,gOri,attr in zip(gazeVecs,gazeOrigins,attrs):
        if gVec is None or gOri is None:
            continue
        # get gaze vector and point on vector (origin, e.g. pupil center) ->
        # transform from ET data coordinate frame into camera coordinate frame
        gVec    = np.matmul(RCam ,          gVec    )
        gOri    = np.matmul(RtCam,np.append(gOri,1.))
        setattr(gazeWorld,attr[0],gOri)

        # intersect with poster -> yield point on poster in camera reference frame
        gPoster = pose.vectorIntersect(gVec, gOri)
        setattr(gazeWorld,attr[1],gPoster)

        # transform intersection with poster from camera space to poster space
        (x,y,z)  = pose.camToWorld(gPoster)  # z should be very close to zero
        setattr(gazeWorld,attr[2],np.asarray([x, y]))

    return gazeWorld