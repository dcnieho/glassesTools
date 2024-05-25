import numpy as np
import pandas as pd
import math
import cv2
from collections import defaultdict

from . import data_files, drawing, utils

class Gaze:
    _columns_compressed = {'timestamp': 1, 'frame_idx':1,
                           'gazePosCam_vidPos_ray':3,'gazePosCam_vidPos_homography':3,'gazePosCamWorld':3,'gazeOriCamLeft':3,'gazePosCamLeft':3,'gazeOriCamRight':3,'gazePosCamRight':3,
                           'gazePosPlane2D_vidPos_ray':2,'gazePosPlane2D_vidPos_homography':2,'gazePosPlane2DWorld':2,'gazePosPlane2DLeft':2,'gazePosPlane2DRight':2}

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
    def readFromFile(fileName,start=None,end=None):
        readSubset  = start is not None and end is not None
        df          = pd.read_csv(str(fileName), delimiter='\t', index_col=False, dtype=defaultdict(lambda: float, frame_idx=int))
        if readSubset:
            df = df[(df['frame_idx'] >= start) & (df['frame_idx'] <= end)]

        # group columns into numpy arrays, insert None if missing
        cols = [col for col in Gaze._columns_compressed if Gaze._columns_compressed[col]>1]
        allCols = tuple([c for c in df.columns if col in c] for col in cols)
        for c,ac in zip(cols,allCols):
            if ac:
                df[c] = [x for x in df[ac].values]  # make list of numpy arrays
            else:
                df[c] = None

        # keep only the columns we want (this also puts them in the right order even if that doesn't matter since we use kwargs to construct objects)
        df = df[Gaze._columns_compressed.keys()]

        # make the gaze objects
        gaze = [Gaze(**kwargs) for kwargs in df.to_dict(orient='records')]

        # organize into dict by frame index
        gazes = {}
        for k,v in zip(df['frame_idx'],gaze):
            gazes.setdefault(k, []).append(v)

        return gazes

    @staticmethod
    def writeToFile(gazes: list['Gaze'], fileName, skip_missing=False):
        data_files.write_array_to_file(gazes, fileName,
                                       Gaze._columns_compressed,
                                       skip_all_nan=skip_missing)

    def drawOnWorldVideo(self, img, cameraMatrix, distCoeff, subPixelFac=1):
        # project to camera, display
        # gaze ray
        if self.gazePosCam_vidPos_ray is not None:
            pPointCam = cv2.projectPoints(self.gazePosCam_vidPos_ray.reshape(1,3),np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            drawing.openCVCircle(img, pPointCam, 3, (255,255,0), -1, subPixelFac)
        # binocular gaze point
        if self.gazePosCamWorld is not None:
            pPointCam = cv2.projectPoints(self.gazePosCamWorld.reshape(1,3),np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            drawing.openCVCircle(img, pPointCam, 3, (255,0,255), -1, subPixelFac)
        # left eye
        if self.gazePosCamLeft is not None:
            pPointCam = cv2.projectPoints(self.gazePosCamLeft.reshape(1,3),np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            drawing.openCVCircle(img, pPointCam, 3, (0,0,255), -1, subPixelFac)
        # right eye
        if self.gazePosCamRight is not None:
            pPointCam = cv2.projectPoints(self.gazePosCamRight.reshape(1,3),np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            drawing.openCVCircle(img, pPointCam, 3, (255,0,0), -1, subPixelFac)
        # average
        if (self.gazePosCamLeft is not None) and (self.gazePosCamRight is not None):
            pointCam  = np.array([(x+y)/2 for x,y in zip(self.gazePosCamLeft,self.gazePosCamRight)]).reshape(1,3)
            pPointCam = cv2.projectPoints(pointCam,np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            if not math.isnan(pPointCam[0]):
                drawing.openCVCircle(img, pPointCam, 6, (255,0,255), -1, subPixelFac)

    def drawOnPoster(self, img, reference, subPixelFac=1):
        # binocular gaze point
        if self.gazePosPlane2DWorld is not None:
            reference.draw(img, self.gazePosPlane2DWorld[0],self.gazePosPlane2DWorld[1], subPixelFac, (0,255,255), 3)
        # left eye
        if self.gazePosPlane2DLeft is not None:
            reference.draw(img, self.gazePosPlane2DLeft[0],self.gazePosPlane2DLeft[1], subPixelFac, (0,0,255), 3)
        # right eye
        if self.gazePosPlane2DRight is not None:
            reference.draw(img, self.gazePosPlane2DRight[0],self.gazePosPlane2DRight[1], subPixelFac, (255,0,0), 3)
        # average
        if (self.gazePosPlane2DLeft is not None) and (self.gazePosPlane2DRight is not None):
            average = np.array([(x+y)/2 for x,y in zip(self.gazePosPlane2DLeft,self.gazePosPlane2DRight)])
            if not math.isnan(average[0]):
                reference.draw(img, average[0], average[1], subPixelFac, (255,0,255))
        # video gaze position
        if self.gazePosPlane2D_vidPos_homography is not None:
            reference.draw(img, self.gazePosPlane2D_vidPos_homography[0],self.gazePosPlane2D_vidPos_homography[1], subPixelFac, (0,255,0), 5)
        if self.gazePosPlane2D_vidPos_ray is not None:
            reference.draw(img, self.gazePosPlane2D_vidPos_ray[0],self.gazePosPlane2D_vidPos_ray[1], subPixelFac, (255,255,0), 3)