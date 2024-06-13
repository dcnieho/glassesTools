import numpy as np
import pandas as pd
import cv2
import pathlib
import math

from . import data_files, drawing, marker, ocv


class Plane:
    default_ref_image_name = 'reference_image.png'

    def __init__(self,
                 markers                : str|pathlib.Path|pd.DataFrame,                            # if str or Path: file from which to read markers. Else direction N_markerx4 array. Should contain centers of markers
                 marker_size            : float,                                                    # in "unit" units

                 aruco_dict                                             = cv2.aruco.DICT_4X4_250,
                 marker_border_bits                                     = 1,
                 marker_pos_scale_fac                                   = 1.,                       # scale factor for marker positions in the markers input argument
                 unit                   : str                           = None,                     # Unit in which measurements (marker size and positions for instance) are expressed. Purely informational
                 package_to_read_from   : str                           = None,                     # if provided, reads marker file from specified package's resources
                 ref_image_store_path   : str|pathlib.Path              = None,
                 ref_image_width                                        = 1920
                 ):

        self.marker_size                                    = marker_size
        # marker positions
        self.markers            : dict[int,marker.Marker]   = {}
        self.bbox               : list[float]               = []

        # marker specs
        self.aruco_dict                                     = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.marker_border_bits                             = marker_border_bits
        self.unit                                           = unit

        # prep markers
        self._get_markers(markers, marker_pos_scale_fac, package_to_read_from)

        # get reference image of plane
        if ref_image_store_path:
            ref_image_store_path = pathlib.Path(ref_image_store_path)

        # get image
        img = None
        # read from file if image exists
        if ref_image_store_path is not None and ref_image_store_path.is_file():
            img = cv2.imread(ref_image_store_path, cv2.IMREAD_COLOR)
        # if image doesn't exist or is the wrong size, create
        if img is None or img.shape[1]!=ref_image_width:
            img = self._store_reference_image(ref_image_store_path, ref_image_width)

        self._ref_image_width                               = ref_image_width
        self._ref_image_cache   : dict[int, np.ndarray]     = {ref_image_width: img}

    def set_center(self, center: np.ndarray):
        # set center of plane. Center coordinate is on current (not original) plane
        # so set_center([5., 0.]) three times in a row shift the center rightward by 15 units
        for i in self.markers:
            self.markers[i].shift(-center)

        self.bbox[0] -= center[0]
        self.bbox[2] -= center[0]
        self.bbox[1] -= center[1]
        self.bbox[3] -= center[1]

    def get_ref_image(self, width: int=None, asRGB=False):
        if width is None:
            width = self._ref_image_width
        assert isinstance(width,int), f'width input should be an int, not {type(width)}'
        # check we have the image, if not, add to cache
        if width not in self._ref_image_cache:
            scale = float(width)/self._ref_image_width
            self._ref_image_cache[width] = cv2.resize(self._ref_image_cache[self._ref_image_width], None, fx=scale, fy=scale, interpolation = cv2.INTER_AREA)

        # return
        if asRGB:
            return self._ref_image_cache[width][:,:,[2,1,0]]    # indexing returns a copy
        else:
            # OpenCV's BGR
            return self._ref_image_cache[width].copy()

    def draw(self, img, x, y, subPixelFac=1, color=None, size=6):
        from . import transforms
        if not math.isnan(x):
            xy = transforms.toImagePos(x, y, self.bbox, [img.shape[1], img.shape[0]])
            if color is None:
                drawing.openCVCircle(img, xy, 8, (0,255,0), -1, subPixelFac)
                color = (0,0,0)
            drawing.openCVCircle(img, xy, size, color, -1, subPixelFac)

    def _get_markers(self, markers: str|pathlib.Path|pd.DataFrame, marker_pos_scale_fac: float, package_to_read_from: str|None):
        # read in aruco marker positions
        markerHalfSizeMm  = self.marker_size/2.

        if isinstance(markers, pd.DataFrame):
            marker_pos = markers
        else:
            marker_pos = data_files.read_coord_file(markers, package_to_read_from)
        if marker_pos is None:
            return

        # turn into marker objects
        marker_pos.x = marker_pos.x.astype('float32')*marker_pos_scale_fac
        marker_pos.y = marker_pos.y.astype('float32')*marker_pos_scale_fac

        for idx, row in marker_pos.iterrows():
            c   = row[['x','y']].values
            # rotate markers (negative because plane coordinate system)
            rot = row[['rotation_angle']].values[0]
            if rot%90 != 0:
                raise ValueError("Rotation of a marker must be a multiple of 90 degrees")
            rotr= -math.radians(rot)
            R   = np.array([[math.cos(rotr), math.sin(rotr)], [-math.sin(rotr), math.cos(rotr)]])
            # top left first, and clockwise: same order as detected ArUco marker corners
            tl = c + np.matmul(R, np.array([-markerHalfSizeMm, -markerHalfSizeMm]))
            tr = c + np.matmul(R, np.array([ markerHalfSizeMm, -markerHalfSizeMm]))
            br = c + np.matmul(R, np.array([ markerHalfSizeMm,  markerHalfSizeMm]))
            bl = c + np.matmul(R, np.array([-markerHalfSizeMm,  markerHalfSizeMm]))

            self.markers[idx] = marker.Marker(idx, c, corners=[tl, tr, br, bl], rot=rot)

        # determine bounding box of markers ([left, top, right, bottom])
        all_x = [c[0] for idx in self.markers for c in self.markers[idx].corners]
        all_y = [c[1] for idx in self.markers for c in self.markers[idx].corners]
        self.bbox.append(min(all_x))
        self.bbox.append(min(all_y))
        self.bbox.append(max(all_x))
        self.bbox.append(max(all_y))

    def get_aruco_board(self, unrotate_markers=False):
        from . import aruco
        board_corner_points = []
        ids = []
        for key in self.markers:
            ids.append(key)
            marker_corner_points = np.vstack(self.markers[key].corners).astype('float32')
            if unrotate_markers:
                marker_corner_points = marker.getUnrotated(marker_corner_points, self.markers[key].rot)

            board_corner_points.append(marker_corner_points)
        return aruco.create_board(board_corner_points, ids, self.aruco_dict)

    def _store_reference_image(self, path: pathlib.Path, width: int) -> np.ndarray:
        referenceBoard = self.get_aruco_board(unrotate_markers = True)
        # get image with markers
        bboxExtents = [self.bbox[2]-self.bbox[0], math.fabs(self.bbox[3]-self.bbox[1])]  # math.fabs to deal with bboxes where (-,-) is bottom left
        aspectRatio = bboxExtents[0]/bboxExtents[1]
        height      = math.ceil(width/aspectRatio)
        margin      = 1  # always 1 pixel, anything else behaves strangely (markers are drawn over margin as well)

        img  = cv2.cvtColor(
            referenceBoard.generateImage(
                (width+2*margin,height+2*margin),margin,self.marker_border_bits),
            cv2.COLOR_GRAY2RGB
        )
        # cut off this 1-pix margin
        assert img.shape[0]==height+2*margin,"Output image height is not as expected"
        assert img.shape[1]==width +2*margin,"Output image width is not as expected"
        img  = img[1:-1,1:-1,:]
        # walk through all markers, if any are supposed to be rotated, do so
        minX =  np.inf
        maxX = -np.inf
        minY =  np.inf
        maxY = -np.inf
        rots = []
        corner_points_unrot = []
        for key in self.markers:
            corner_points = np.vstack(self.markers[key].corners).astype('float32')
            corner_points_unrot.append(marker.getUnrotated(corner_points, self.markers[key].rot))
            rots.append(self.markers[key].rot)
            minX = np.min(np.hstack((minX,corner_points[:,0])))
            maxX = np.max(np.hstack((maxX,corner_points[:,0])))
            minY = np.min(np.hstack((minY,corner_points[:,1])))
            maxY = np.max(np.hstack((maxY,corner_points[:,1])))
        if np.any(np.array(rots)!=0):
            # determine where the markers are placed
            sizeX = maxX - minX
            sizeY = maxY - minY
            xReduction = sizeX / float(img.shape[1])
            yReduction = sizeY / float(img.shape[0])
            if xReduction > yReduction:
                nRows = int(sizeY / xReduction);
                yMargin = (img.shape[0] - nRows) / 2;
                xMargin = 0
            else:
                nCols = int(sizeX / yReduction);
                xMargin = (img.shape[1] - nCols) / 2;
                yMargin = 0

            for r,cpu in zip(rots,corner_points_unrot):
                if r != 0:
                    # figure out where marker is
                    cpu -= np.array([[minX,minY]])
                    cpu[:,0] =       cpu[:,0] / sizeX  * float(img.shape[1]) + xMargin
                    cpu[:,1] = (1. - cpu[:,1] / sizeY) * float(img.shape[0]) + yMargin
                    sz = np.min(cpu[2,:]-cpu[0,:])
                    # get marker
                    cpu = np.floor(cpu)
                    idxs = np.floor([cpu[0,1], cpu[0,1]+sz, cpu[0,0], cpu[0,0]+sz]).astype('int')
                    mark = img[idxs[0]:idxs[1], idxs[2]:idxs[3]]
                    # rotate (opposite because coordinate system) and put back
                    if r==-90:
                        mark = cv2.rotate(mark, cv2.ROTATE_90_CLOCKWISE)
                    elif r==90:
                        mark = cv2.rotate(mark, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif r==180:
                        mark = cv2.rotate(mark, cv2.ROTATE_180)

                    img[idxs[0]:idxs[1], idxs[2]:idxs[3]] = mark

        if path:
            cv2.imwrite(path, img)

        return img


class Pose:
    # description of tsv file used for storage
    _columns_compressed = {'frame_idx':1,
                           'pose_N_markers': 1, 'pose_R_vec': 3, 'pose_T_vec': 3,
                           'homography_N_markers': 1, 'homography_mat': 9}
    _non_float          = {'frame_idx': int, 'pose_ok': bool, 'pose_N_markers': int, 'homography_N_markers': int}

    def __init__(self,
                 frame_idx              : int,
                 pose_N_markers         : int       = 0,
                 pose_R_vec             : np.ndarray= None,
                 pose_T_vec             : np.ndarray= None,
                 homography_N_markers   : int       = 0,
                 homography_mat         : np.ndarray= None):
        self.frame_idx            : int         = frame_idx
        # pose
        self.pose_N_markers       : int         = pose_N_markers        # number of ArUco markers this pose estimate is based on. 0 if failed
        self.pose_R_vec           : np.ndarray  = pose_R_vec
        self.pose_T_vec           : np.ndarray  = pose_T_vec
        # homography
        self.homography_N_markers : int         = homography_N_markers  # number of ArUco markers this homongraphy estimate is based on. 0 if failed
        self.homography_mat       : np.ndarray  = homography_mat.reshape(3,3) if homography_mat is not None else homography_mat

        # internals
        self._RMat        = None
        self._RtMat       = None
        self._planeNormal = None
        self._planePoint  = None
        self._RMatInv     = None
        self._RtMatInv    = None
        self._iH          = None

    def camToWorld(self, point: np.ndarray):
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(point)):
            return np.full((3,), np.nan)

        if self._RtMatInv is None:
            if self._RMatInv is None:
                if self._RMat is None:
                    self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
                self._RMatInv = self._RMat.T
            self._RtMatInv = np.hstack((self._RMatInv,np.matmul(-self._RMatInv,self.pose_T_vec.reshape(3,1))))

        return np.matmul(self._RtMatInv,np.append(np.array(point),1.).reshape((4,1))).flatten()

    def worldToCam(self, point: np.ndarray):
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(point)):
            return np.full((3,), np.nan)

        if self._RtMat is None:
            if self._RMat is None:
                self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
            self._RtMat = np.hstack((self._RMat, self.pose_T_vec.reshape(3,1)))

        return np.matmul(self._RtMat,np.append(np.array(point),1.).reshape((4,1))).flatten()

    def planeToCamPose(self, point: np.ndarray, camera_params: ocv.CameraParams) -> np.ndarray:
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(point)) or not camera_params.has_intrinsics():
            return np.full((2,), np.nan)
        return cv2.projectPoints(point, self.pose_R_vec, self.pose_T_vec, camera_params.camera_mtx, camera_params.distort_coeffs)[0].flatten()

    def camToPlanePose(self, point: np.ndarray, camera_params: ocv.CameraParams) -> np.ndarray:
        # NB: also returns intermediate result (intersection with poster in camera space)
        from . import transforms
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(point)) or not camera_params.has_intrinsics():
            return np.full((2,), np.nan), np.full((3,), np.nan)

        g3D = transforms.unprojectPoint(*point, camera_params.camera_mtx, camera_params.distort_coeffs)

        # find intersection of 3D gaze with poster
        pos_cam = self.vectorIntersect(g3D)  # default vec origin (0,0,0) because we use g3D from camera's view point

        # above intersection is in camera space, turn into poster space to get position on poster
        (x,y,z) = self.camToWorld(pos_cam) # z should be very close to zero
        return np.asarray([x, y]), pos_cam

    def planeToCamHomography(self, point: np.ndarray, camera_params: ocv.CameraParams) -> np.ndarray:
        # from location on plane (2D) to location on camera image (2D)
        from . import transforms
        if self.homography_mat is None:
            return np.full((2,), np.nan)

        if self._iH is None:
            self._iH = np.linalg.inv(self.homography_mat)
        out = transforms.applyHomography(self._iH, *point)
        if camera_params.has_intrinsics():
            out = transforms.distortPoint(*out, camera_params.camera_mtx, camera_params.distort_coeffs)
        return out

    def camToPlaneHomography(self, point: np.ndarray, camera_params: ocv.CameraParams) -> np.ndarray:
        # from location on camera image (2D) to location on plane (2D)
        from . import transforms
        if self.homography_mat is None:
            return np.full((2,), np.nan)

        if camera_params.has_intrinsics():
            point = transforms.undistortPoint(*point, camera_params.camera_mtx, camera_params.distort_coeffs)
        return transforms.applyHomography(self.homography_mat, *point)

    def getOriginOnImage(self, camera_params: ocv.CameraParams) -> np.ndarray:
        if self.pose_N_markers>0 and camera_params.has_intrinsics():
            a = cv2.projectPoints(np.zeros((1,3)), self.pose_R_vec,self.pose_T_vec, camera_params.camera_mtx,camera_params.distort_coeffs)[0].flatten()
        elif self.homography_N_markers>0:
            a = self.planeToCamHomography([0., 0.], camera_params.camera_mtx, camera_params.distort_coeffs)
        else:
            a = np.full((2,), np.nan)
        return a

    def vectorIntersect(self, vector: np.ndarray, origin = np.array([0.,0.,0.])):
        from . import transforms

        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(vector)):
            return np.full((3,), np.nan)

        if self._planeNormal is None:
            if self._RtMat is None:
                if self._RMat is None:
                    self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
                self._RtMat = np.hstack((self._RMat, self.pose_T_vec.reshape(3,1)))

            # get poster normal
            self._planeNormal = np.matmul(self._RMat, np.array([0., 0., 1.]))
            # get point on poster (just use origin)
            self._planePoint  = np.matmul(self._RtMat, np.array([0., 0., 0., 1.]))

        # normalize vector
        vector /= np.sqrt((vector**2).sum())

        # find intersection of 3D gaze with poster
        return transforms.intersect_plane_ray(self._planeNormal, self._planePoint, vector.flatten(), origin.flatten())


def read_dict_from_file(fileName:str|pathlib.Path, episodes:list[list[int]]=None) -> dict[int,Pose]:
    return data_files.read_file(fileName,
                                Pose, True, True, False, False,
                                episodes=episodes)[0]

def write_list_to_file(poses: list[Pose], fileName:str|pathlib.Path, skip_failed=False):
    data_files.write_array_to_file(poses, fileName,
                                    Pose._columns_compressed,
                                    skip_all_nan=skip_failed)