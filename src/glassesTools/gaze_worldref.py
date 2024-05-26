import numpy as np
import math
import cv2
import pathlib

from . import data_files, drawing

class Gaze:
    _columns_compressed = {'timestamp': 1, 'frame_idx':1,
                           'gazePosCam_vidPos_ray':3,'gazePosCam_vidPos_homography':3,'gazePosCamWorld':3,'gazeOriCamLeft':3,'gazePosCamLeft':3,'gazeOriCamRight':3,'gazePosCamRight':3,
                           'gazePosPlane2D_vidPos_ray':2,'gazePosPlane2D_vidPos_homography':2,'gazePosPlane2DWorld':2,'gazePosPlane2DLeft':2,'gazePosPlane2DRight':2}
    _non_float          = {'frame_idx': int}

    def __init__(self, timestamp, frame_idx,
                 gazePosCam_vidPos_ray=None, gazePosCam_vidPos_homography=None, gazePosCamWorld=None, gazeOriCamLeft=None, gazePosCamLeft=None, gazeOriCamRight=None, gazePosCamRight=None,
                 gazePosPlane2D_vidPos_ray=None, gazePosPlane2D_vidPos_homography=None, gazePosPlane2DWorld=None, gazePosPlane2DLeft=None, gazePosPlane2DRight=None):
        # 3D gaze is in world space, w.r.t. scene camera
        # 2D gaze is on the poster
        self.timestamp                          = timestamp
        self.frame_idx                          = frame_idx

        # in camera space (3D coordinates)
        self.gazePosCam_vidPos_ray              = gazePosCam_vidPos_ray             # video gaze position on plane (camera ray intersected with plane)
        self.gazePosCam_vidPos_homography       = gazePosCam_vidPos_homography      # gaze2DHomography in camera space
        self.gazePosCamWorld                    = gazePosCamWorld                   # 3D gaze point on plane (world-space gaze point, turned into direction ray and intersected with plane)
        self.gazeOriCamLeft                     = gazeOriCamLeft
        self.gazePosCamLeft                     = gazePosCamLeft                    # 3D gaze point on plane ( left eye gaze vector intersected with plane)
        self.gazeOriCamRight                    = gazeOriCamRight
        self.gazePosCamRight                    = gazePosCamRight                   # 3D gaze point on plane (right eye gaze vector intersected with plane)

        # in poster space (2D coordinates)
        self.gazePosPlane2D_vidPos_ray          = gazePosPlane2D_vidPos_ray        # Video gaze point mapped to poster by turning into direction ray and intersecting with poster
        self.gazePosPlane2D_vidPos_homography   = gazePosPlane2D_vidPos_homography # Video gaze point directly mapped to poster through homography transformation
        self.gazePosPlane2DWorld                = gazePosPlane2DWorld              # wGaze3D in poster space
        self.gazePosPlane2DLeft                 = gazePosPlane2DLeft               # lGaze3D in poster space
        self.gazePosPlane2DRight                = gazePosPlane2DRight              # rGaze3D in poster space

    @staticmethod
    def readFromFile(fileName:str|pathlib.Path, start:int=None, end:int=None) -> dict[int,list['Gaze']]:
        return data_files.read_file(fileName,
                                    Gaze, False, False, True,
                                    start=start,end=end)[0]

    @staticmethod
    def writeToFile(gazes: list['Gaze'], fileName, skip_missing=False):
        data_files.write_array_to_file(gazes, fileName,
                                       Gaze._columns_compressed,
                                       skip_all_nan=skip_missing)

    def drawOnWorldVideo(self, img, cameraMatrix, distCoeff, subPixelFac=1):
        # project to camera, display
        def project_and_draw(img,pos,sz,clr,subPixelFac):
            pPointCam = cv2.projectPoints(pos.reshape(1,3),np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            if not math.isnan(pPointCam[0]):
                drawing.openCVCircle(img, pPointCam, sz, clr, -1, subPixelFac)

        # gaze ray
        if self.gazePosCam_vidPos_ray is not None:
            project_and_draw(img, self.gazePosCam_vidPos_ray, 3, (255,255,0), subPixelFac)
        # binocular gaze point
        if self.gazePosCamWorld is not None:
            project_and_draw(img, self.gazePosCamWorld, 3, (255,0,255), subPixelFac)
        # left eye
        if self.gazePosCamLeft is not None:
            project_and_draw(img, self.gazePosCamLeft, 3, (0,0,255), subPixelFac)
        # right eye
        if self.gazePosCamRight is not None:
            project_and_draw(img, self.gazePosCamRight, 3, (255,0,0), subPixelFac)
        # average
        if (self.gazePosCamLeft is not None) and (self.gazePosCamRight is not None):
            pointCam  = np.array([(x+y)/2 for x,y in zip(self.gazePosCamLeft,self.gazePosCamRight)])
            project_and_draw(img, pointCam, 6, (255,0,255), subPixelFac)

    def drawOnPoster(self, img, reference, subPixelFac=1):
        # binocular gaze point
        if self.gazePosPlane2DWorld is not None:
            reference.draw(img, *self.gazePosPlane2DWorld, subPixelFac, (0,255,255), 3)
        # left eye
        if self.gazePosPlane2DLeft is not None:
            reference.draw(img, *self.gazePosPlane2DLeft, subPixelFac, (0,0,255), 3)
        # right eye
        if self.gazePosPlane2DRight is not None:
            reference.draw(img, *self.gazePosPlane2DRight, subPixelFac, (255,0,0), 3)
        # average
        if (self.gazePosPlane2DLeft is not None) and (self.gazePosPlane2DRight is not None):
            average = np.array([(x1+x2)/2 for x1,x2 in zip(self.gazePosPlane2DLeft,self.gazePosPlane2DRight)])
            if not math.isnan(average[0]):
                reference.draw(img, *average, subPixelFac, (255,0,255))
        # video gaze position
        if self.gazePosPlane2D_vidPos_homography is not None:
            reference.draw(img, *self.gazePosPlane2D_vidPos_homography, subPixelFac, (0,255,0), 5)
        if self.gazePosPlane2D_vidPos_ray is not None:
            reference.draw(img, *self.gazePosPlane2D_vidPos_ray, subPixelFac, (255,255,0), 3)