import numpy as np
import math
import cv2

from . import marker

def openCVCircle(img, center_coordinates, radius, color, thickness, sub_pixel_fac):
    p = [np.round(x*sub_pixel_fac) for x in center_coordinates]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(np.intc).max for x in p]):
        p = tuple([int(x) for x in p])
        cv2.circle(img, p, radius*sub_pixel_fac, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(sub_pixel_fac)))

def openCVLine(img, start_point, end_point, color, thickness, sub_pixel_fac):
    sp = [np.round(x*sub_pixel_fac) for x in start_point]
    ep = [np.round(x*sub_pixel_fac) for x in   end_point]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(np.intc).max for x in sp]) and np.all([not math.isnan(x) and abs(x)<np.iinfo(np.intc).max for x in ep]):
        sp = tuple([int(x) for x in sp])
        ep = tuple([int(x) for x in ep])
        cv2.line(img, sp, ep, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(sub_pixel_fac)))

def openCVRectangle(img, p1, p2, color, thickness, sub_pixel_fac):
    p1 = [np.round(x*sub_pixel_fac) for x in p1]
    p2 = [np.round(x*sub_pixel_fac) for x in p2]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(np.intc).max for x in p1]) and np.all([not math.isnan(x) and abs(x)<np.iinfo(np.intc).max for x in p2]):
        p1 = tuple([int(x) for x in p1])
        p2 = tuple([int(x) for x in p2])
        cv2.rectangle(img, p1, p2, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(sub_pixel_fac)))

def openCVFrameAxis(img, camera_matrix, dist_coeffs, rvec,  tvec,  arm_length, thickness, sub_pixel_fac, position = [0.,0.,0.]):
    # same as the openCV function, but with anti-aliasing for a nicer image if subPixelFac>1
    points = np.vstack((np.zeros((1,3)), arm_length*np.eye(3)))+np.vstack(4*[np.asarray(position)])
    cam_points = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)[0]
    # z-sort them
    RMat = cv2.Rodrigues(rvec)[0]
    RtMat = np.hstack((RMat, tvec.reshape(3,1)))
    world_points = np.matmul(RtMat,np.pad(points[1:,:],((0,0),(0,1)),'constant', constant_values=(1.,1.)).T)
    order = np.argsort(world_points[-1,:])[::-1]
    # draw
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    for i in order:
        openCVLine(img, cam_points[0].flatten(), cam_points[i+1].flatten(), colors[i], thickness, sub_pixel_fac)

def arucoDetectedMarkers(img,corners,ids,border_color=(0,255,0), draw_IDs = True, sub_pixel_fac=1, special_highlight = None):
    if special_highlight is None:
        special_highlight = []
    # same as the openCV function, but with anti-aliasing for a (much) nicer image if subPixelFac>1
    # and ability to use a different color from some of the markers
    textColor   = [x for x in border_color]
    cornerColor = [x for x in border_color]
    textColor[0]  , textColor[1]   = textColor[1]  , textColor[0]       #   text color just swap B and R
    cornerColor[1], cornerColor[2] = cornerColor[2], cornerColor[1]     # corner color just swap G and R

    draw_IDs = draw_IDs and (ids is not None) and len(ids)>0

    for i in range(0, len(corners)):
        corner = corners[i][0]
        # draw marker sides
        sideColor = border_color
        for s,c in zip(special_highlight[::2],special_highlight[1::2]):
            if s is not None and ids[i][0] in s:
                sideColor = c
        for j in range(4):
            p0 = corner[j,:]
            p1 = corner[(j + 1) % 4,:]
            openCVLine(img, p0, p1, sideColor, 1, sub_pixel_fac)

        # draw first corner mark
        p1 = corner[0]
        openCVRectangle(img, corner[0]-3, corner[0]+3, cornerColor, 1, sub_pixel_fac)

        # draw IDs if wanted
        if draw_IDs:
            c = marker.corners_intersection(corner)
            cv2.putText(img, str(ids[i][0]), tuple(c.astype(np.intc)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2, lineType=cv2.LINE_AA)