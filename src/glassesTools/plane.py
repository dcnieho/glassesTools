import numpy as np
import pandas as pd
import cv2
import pathlib
import math
import typing


from . import data_files, drawing, marker, transforms

class Coordinate(typing.NamedTuple):
    x: float = 0.
    y: float = 0.

class Plane:
    default_ref_image_name = 'reference_image.png'

    def __init__(self,
                 markers                : str|pathlib.Path|pd.DataFrame,                            # if str or Path: file from which to read markers. Else direction N_markerx4 array. Should contain centers of markers
                 marker_size            : float,                                                    # in "unit" units
                 plane_size             : Coordinate,                                               # in "unit" units

                 aruco_dict_id                                          = cv2.aruco.DICT_4X4_250,
                 marker_border_bits                                     = 1,
                 marker_pos_scale_fac                                   = 1.,                       # scale factor for marker positions in the markers input argument
                 unit                   : str                           = None,                     # Unit in which measurements (marker size and positions for instance) are expressed. Purely informational
                 package_to_read_from   : str                           = None,                     # if provided, reads marker file from specified package's resources
                 ref_image_store_path   : str|pathlib.Path              = None,
                 ref_image_size         : int                           = 1920,                     # largest dimension
                 min_num_markers        : int                           = 3                         # minimum number of markers for gaze to be mapped to this plane
                 ):

        self.marker_size                                    = marker_size
        # marker positions
        self.markers            : dict[int,marker.Marker]   = {}
        self._all_marker_ids    : list[int]                 = []
        self.plane_size                                     = plane_size
        self.bbox               : list[float]               = [0., 0., self.plane_size.x, self.plane_size.y]
        self._origin            : Coordinate                = Coordinate(0., 0.)

        # marker specs
        self.aruco_dict_id                                  = aruco_dict_id
        self.aruco_dict                                     = cv2.aruco.getPredefinedDictionary(self.aruco_dict_id)
        self.marker_border_bits                             = marker_border_bits
        self.unit                                           = unit

        # processing specs
        self.min_num_markers                                = min_num_markers

        # prep markers
        self._load_markers(markers, marker_pos_scale_fac, package_to_read_from)

        # get reference image of plane
        if ref_image_store_path:
            ref_image_store_path = pathlib.Path(ref_image_store_path)

        # get image (always create reference image, to be safe (settings may have changed))
        img = self._store_reference_image(ref_image_store_path, ref_image_size)

        self._ref_image_size                                = ref_image_size
        self._ref_image_cache   : dict[int, np.ndarray]     = {ref_image_size: img}

    def set_origin(self, origin: Coordinate):
        # change from current origin
        offset = np.array(origin)-self._origin

        for i in self.markers:
            self.markers[i].shift(-offset)

        self.bbox[0] -= offset[0]
        self.bbox[2] -= offset[0]
        self.bbox[1] -= offset[1]
        self.bbox[3] -= offset[1]

        self._origin = origin

    def get_ref_image(self, im_size: int=None, as_RGB=False) -> np.ndarray:
        if im_size is None:
            im_size = self._ref_image_size
        if not isinstance(im_size,int):
            raise TypeError(f'width input should be an int, not {type(im_size)}')
        # check we have the image, if not, add to cache
        if im_size not in self._ref_image_cache:
            scale = float(im_size)/self._ref_image_size
            self._ref_image_cache[im_size] = cv2.resize(self._ref_image_cache[self._ref_image_size], None, fx=scale, fy=scale, interpolation = cv2.INTER_AREA)

        # return
        if as_RGB:
            return self._ref_image_cache[im_size][:,:,[2,1,0]].copy()
        else:
            # OpenCV's BGR
            return self._ref_image_cache[im_size].copy()

    def draw(self, img, x, y, sub_pixel_fac=1, color=None, size=6):
        if not math.isnan(x):
            xy = transforms.to_image_pos(x, y, self.bbox, [img.shape[1], img.shape[0]])
            if color is None:
                drawing.openCVCircle(img, xy, 8, (0,255,0), -1, sub_pixel_fac)
                color = (0,0,0)
            drawing.openCVCircle(img, xy, size, color, -1, sub_pixel_fac)

    def _load_markers(self, markers: str|pathlib.Path|pd.DataFrame, marker_pos_scale_fac: float, package_to_read_from: str|None):
        from . import aruco
        # read in aruco marker positions
        markerHalfSizeMm  = self.marker_size/2.

        if isinstance(markers, pd.DataFrame):
            marker_pos = markers
        else:
            marker_pos = data_files.read_coord_file(markers, package_to_read_from)
        if marker_pos is None:
            raise RuntimeError(f"No markers could be read from the file {markers}, check it exists and contains markers")

        # keep track of all IDs so we can check for duplicates
        self._all_marker_ids = marker_pos.index.to_list()
        marker_dict_size = aruco.get_dict_size(self.aruco_dict_id)
        for m_id in self._all_marker_ids:
            if m_id>=marker_dict_size:
                raise ValueError(f'This plane is set up using the dictionary {aruco.dict_id_to_str[self.aruco_dict_id]} which only has {marker_dict_size} markers, which means that valid IDs are 0-{marker_dict_size-1}. However, this plane is configured to contain a marker number {m_id} that is not a valid marker for this dictionary.')

        # turn into marker objects
        marker_pos.x *= marker_pos_scale_fac
        marker_pos.y *= marker_pos_scale_fac

        for idx, row in marker_pos.iterrows():
            c   = row[['x','y']].values
            # rotate markers (negative because plane coordinate system)
            rot = row[['rotation_angle']].values[0] if 'rotation_angle' in row else 0.
            rotr= -math.radians(rot)
            R   = np.array([[math.cos(rotr), math.sin(rotr)], [-math.sin(rotr), math.cos(rotr)]])
            # top left first, and clockwise: same order as detected ArUco marker corners
            tl = c + np.matmul(R, np.array([-markerHalfSizeMm, -markerHalfSizeMm]))
            tr = c + np.matmul(R, np.array([ markerHalfSizeMm, -markerHalfSizeMm]))
            br = c + np.matmul(R, np.array([ markerHalfSizeMm,  markerHalfSizeMm]))
            bl = c + np.matmul(R, np.array([-markerHalfSizeMm,  markerHalfSizeMm]))

            self.markers[idx] = marker.Marker(idx, c, corners=[tl, tr, br, bl], rot=rot)

    def get_marker_IDs(self) -> dict[str|int,list[marker.MarkerID]]:
        return {'plane': [marker.MarkerID(m_id, self.aruco_dict_id) for m_id in self._all_marker_ids]}

    def get_aruco_board(self) -> cv2.aruco.Board:
        from . import aruco
        board_corner_points = []
        ids = []
        for key in self.markers:
            ids.append(key)
            marker_corner_points = np.vstack(self.markers[key].corners).astype('float32')
            board_corner_points.append(marker_corner_points)
        return aruco.create_board(board_corner_points, ids, self.aruco_dict)

    def get_plane_setup(self):
        from . import aruco
        return aruco.PlaneSetup(plane = self,
                                aruco_detector_params = {
                                    'markerBorderBits': self.marker_border_bits
                                },
                                min_num_markers = self.min_num_markers)

    def _store_reference_image(self, path: pathlib.Path, im_size: int) -> np.ndarray:
        # get image with markers
        bbox_extents = [self.bbox[2]-self.bbox[0], math.fabs(self.bbox[3]-self.bbox[1])]  # math.fabs to deal with bboxes where (-,-) is bottom left
        aspect_ratio = bbox_extents[0]/bbox_extents[1]
        if aspect_ratio>1:
            width       = im_size
            height      = math.ceil(im_size/aspect_ratio)
        else:
            width       = math.ceil(im_size*aspect_ratio)
            height      = im_size

        img = np.zeros((height, width), np.uint8)
        img[:] = 255
        # collect all markers
        corner_points = []
        ids = []
        # for checking if markers fit
        x_margin = bbox_extents[0]/width /5     # ignore .2 pixel or less
        y_margin = bbox_extents[1]/height/5     # ignore .2 pixel or less
        for key in self.markers:
            ids.append(key)
            corners = np.vstack(self.markers[key].corners).astype('float32')
            # check we're on the plane
            if np.any(corners[:,0]<-x_margin) or np.any(corners[:,0]>self.plane_size.x+x_margin) or \
               np.any(corners[:,1]<-y_margin) or np.any(corners[:,1]>self.plane_size.y+y_margin):
                center  = ", ".join(map(lambda x: f"{x:.4f}",self.markers[key].center))
                corners = [", ".join(map(lambda x: f"{x:.4f}",c)) for c in corners]
                plane_corners = [", ".join(map(lambda x: f"{x:.4f}",c)) for c in (self.bbox[:2],self.bbox[2:])]
                raise ValueError(f'Marker {key} with center positioned at ({center}), size {self.marker_size:.4f} and rotation {self.markers[key].rot:.1f} deg would\nhave its corners at ({corners[0]}), ({corners[1]}), ({corners[2]}), and ({corners[3]}),\nwhich is outside the defined plane which ranges from ({plane_corners[0]}) to ({plane_corners[1]}). Ensure all\nsizes and positions are in the same unit (e.g. mm) and check the marker position csv file, marker size and plane size.')
            corner_points.append(corners)

        # get info about marker positions on the board
        corner_points = np.dstack(corner_points)
        corner_points = corner_points-np.expand_dims(np.array([self.bbox[:2]]),2).astype('float32')

        # get position and size of marker in the generated image
        corner_points[:,0,:] = corner_points[:,0,:]/bbox_extents[0] * float(img.shape[1])
        corner_points[:,1,:] = corner_points[:,1,:]/bbox_extents[1] * float(img.shape[0])

        # get marker size
        pix_sz = np.vstack((np.hypot(corner_points[0,0,:]-corner_points[1,0,:], corner_points[0,1,:]-corner_points[1,1,:]),
                            np.hypot(corner_points[1,0,:]-corner_points[2,0,:], corner_points[1,1,:]-corner_points[2,1,:]))).T
        # marker should be square
        pix_sz = np.round(np.min(pix_sz,1)).astype('int')

        # place markers
        for i,sz,pos in zip(ids,pix_sz,np.moveaxis(corner_points, -1, 0)):
            # make marker
            marker_image = np.zeros((sz, sz), dtype=np.uint8)
            marker_image = self.aruco_dict.generateImageMarker(i, sz, marker_image, self.marker_border_bits)

            # put in image
            if pos[0,1]==pos[1,1] and pos[1,0]==pos[2,0] and pos[0,0]<pos[1,0]:
                # marker is aligned to image axes and not rotated, just blit
                ori = np.round(pos[0,:]).astype('int')
                img[ori[1]:ori[1]+sz, ori[0]:ori[0]+sz] = marker_image
                continue

            # set up affine transformation for placing marker in image
            in_corners = np.array([[-.5, -.5],
                                   [marker_image.shape[1]-.5, -.5],
                                   [marker_image.shape[1]-.5, marker_image.shape[0]-.5]]).astype('float32')
            transformation = cv2.getAffineTransform(in_corners, pos[:3,:])

            # perform affine transformation (i.e. rotate marker)
            img = cv2.warpAffine(marker_image, transformation, (img.shape[1], img.shape[0]), img, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if path:
            cv2.imwrite(path, img)

        return img