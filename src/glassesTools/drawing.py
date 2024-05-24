import numpy as np
import math
import cv2

from . import marker

def openCVCircle(img, center_coordinates, radius, color, thickness, subPixelFac):
    p = [np.round(x*subPixelFac) for x in center_coordinates]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in p]):
        p = tuple([int(x) for x in p])
        cv2.circle(img, p, radius*subPixelFac, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))

def openCVLine(img, start_point, end_point, color, thickness, subPixelFac):
    sp = [np.round(x*subPixelFac) for x in start_point]
    ep = [np.round(x*subPixelFac) for x in   end_point]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in sp]) and np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in ep]):
        sp = tuple([int(x) for x in sp])
        ep = tuple([int(x) for x in ep])
        cv2.line(img, sp, ep, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))

def openCVRectangle(img, p1, p2, color, thickness, subPixelFac):
    p1 = [np.round(x*subPixelFac) for x in p1]
    p2 = [np.round(x*subPixelFac) for x in p2]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in p1]) and np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in p2]):
        p1 = tuple([int(x) for x in p1])
        p2 = tuple([int(x) for x in p2])
        cv2.rectangle(img, p1, p2, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))

def openCVFrameAxis(img, cameraMatrix, distCoeffs, rvec,  tvec,  armLength, thickness, subPixelFac, position = [0.,0.,0.]):
    # same as the openCV function, but with anti-aliasing for a nicer image if subPixelFac>1
    points = np.vstack((np.zeros((1,3)), armLength*np.eye(3)))+np.vstack(4*[np.asarray(position)])
    points = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)[0]
    openCVLine(img, points[0].flatten(), points[1].flatten(), (0, 0, 255), thickness, subPixelFac)
    openCVLine(img, points[0].flatten(), points[2].flatten(), (0, 255, 0), thickness, subPixelFac)
    openCVLine(img, points[0].flatten(), points[3].flatten(), (255, 0, 0), thickness, subPixelFac)

def arucoDetectedMarkers(img,corners,ids,borderColor=(0,255,0), drawIDs = True, subPixelFac=1, specialHighlight = []):
    # same as the openCV function, but with anti-aliasing for a (much) nicer image if subPixelFac>1
    textColor   = [x for x in borderColor]
    cornerColor = [x for x in borderColor]
    textColor[0]  , textColor[1]   = textColor[1]  , textColor[0]       #   text color just swap B and R
    cornerColor[1], cornerColor[2] = cornerColor[2], cornerColor[1]     # corner color just swap G and R

    drawIDs = drawIDs and (ids is not None) and len(ids)>0

    for i in range(0, len(corners)):
        corner = corners[i][0]
        # draw marker sides
        sideColor = borderColor
        for s,c in zip(specialHighlight[::2],specialHighlight[1::2]):
            if s is not None and ids[i][0] in s:
                sideColor = c
        for j in range(4):
            p0 = corner[j,:]
            p1 = corner[(j + 1) % 4,:]
            openCVLine(img, p0, p1, sideColor, 1, subPixelFac)

        # draw first corner mark
        p1 = corner[0]
        openCVRectangle(img, corner[0]-3, corner[0]+3, cornerColor, 1, subPixelFac)

        # draw IDs if wanted
        if drawIDs:
            c = marker.corners_intersection(corner)
            cv2.putText(img, str(ids[i][0]), tuple(c.astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2, lineType=cv2.LINE_AA)