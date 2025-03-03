import cv2
import numpy as np
import pathlib
import importlib.resources
import shutil
import math
import typing
from matplotlib import colors

from .... import aruco, drawing, marker, plane, transforms

def deploy_maker(output_dir: str|pathlib.Path):
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError('the requested directory "%s" does not exist' % output_dir)

    # copy over all files
    for r in ['poster.tex']:
        with importlib.resources.path(__package__, r) as p:
            shutil.copyfile(p, str(output_dir/r))

    deploy_marker_images(output_dir)

def deploy_marker_images(output_dir: str|pathlib.Path):
    from .. import get_validation_setup

    output_dir = pathlib.Path(output_dir) / "all-markers"
    if not output_dir.is_dir():
        output_dir.mkdir()

    # get validation setup
    validationSetup = get_validation_setup()

    # generate and store the markers
    aruco.deploy_marker_images(output_dir, 1000, ValidationPlane.default_aruco_dict, validationSetup['markerBorderBits'])

def deploy_default_pdf(output_file_or_dir: str|pathlib.Path):
    output_file_or_dir = pathlib.Path(output_file_or_dir)
    if output_file_or_dir.is_dir():
        output_file_or_dir = output_file_or_dir / 'poster.pdf'

    with importlib.resources.path(__package__,'poster.pdf') as p:
        shutil.copyfile(p, str(output_file_or_dir))


class ValidationPlane(plane.Plane):
    poster_image_filename = 'referencePoster.png'
    default_aruco_dict    = cv2.aruco.DICT_4X4_250

    def __init__(self, config_dir: str|pathlib.Path|None, validation_config: dict[str,typing.Any]=None, **kwarg):
        from .. import get_markers, get_validation_setup
        # NB: if config_dir is None, the default config will be used

        if config_dir is not None:
            config_dir = pathlib.Path(config_dir)

        # get validation config, if needed
        if validation_config is None:
            validation_config = get_validation_setup(config_dir)
        self.config = validation_config

        # get marker width
        if self.config['mode'] == 'deg':
            self.cell_size_mm = 2.*math.tan(math.radians(.5))*self.config['distance']*10
        else:
            self.cell_size_mm = 10 # 1cm
        markerSize = self.cell_size_mm*self.config['markerSide']

        # get board size
        plane_size = plane.Coordinate(self.config['gridCols']*self.cell_size_mm, self.config['gridRows']*self.cell_size_mm)

        # get targets first, so that they can be drawn on the reference image
        self.targets: dict[int,marker.Marker] = {}
        origin = self._get_targets(config_dir, self.config)

        # call base class
        markers = get_markers(config_dir, self.config['markerPosFile'])
        ref_image_store_path = None
        if 'ref_image_store_path' in kwarg:
            ref_image_store_path = kwarg.pop('ref_image_store_path')
        elif config_dir is not None:
            ref_image_store_path = config_dir / self.poster_image_filename
        super(ValidationPlane, self).__init__(markers, markerSize, plane_size, ValidationPlane.default_aruco_dict, self.config['markerBorderBits'],self.cell_size_mm, "mm", ref_image_store_path=ref_image_store_path, ref_image_size=self.config['referencePosterSize'],**kwarg)

        # set center
        self.set_origin(origin)

    def set_origin(self, origin: plane.Coordinate):
        # set origin of plane. Origin location is on current (not original) plane
        # so set_origin([5., 0.]) three times in a row shifts the origin rightward by 15 units
        for i in self.targets:
            self.targets[i].shift(-np.array(origin))
        super(ValidationPlane, self).set_origin(origin)

    def _get_targets(self, config_dir, validationSetup) -> plane.Coordinate:
        """ poster space: (0,0) is origin (might be center target), (-,-) bottom left """
        from .. import get_targets

        # read in target positions
        targets = get_targets(config_dir, validationSetup['targetPosFile'])
        if targets is not None:
            targets['center'] = list(targets[['x','y']].values)
            targets['center'] *= self.cell_size_mm
            targets = targets.drop(['x','y'], axis=1)
            self.targets = {idx:marker.Marker(idx,**kwargs) for idx,kwargs in zip(targets.index.values,targets.to_dict(orient='records'))}
            origin = plane.Coordinate(*targets.loc[validationSetup['centerTarget']].center.copy())  # NB: need origin in scaled space
        else:
            origin = plane.Coordinate(0.,0.)
        return origin

    def _store_reference_image(self, path: pathlib.Path, width: int) -> np.ndarray:
        # first call superclass method to generate image without targets
        img = super(ValidationPlane, self)._store_reference_image(path, width)
        height = img.shape[0]

        # add targets
        subPixelFac = 8   # for sub-pixel positioning
        for key in self.targets:
            # 1. determine position on image
            circlePos = transforms.to_image_pos(*self.targets[key].center, self.bbox,[width,height])

            # 2. draw
            clr = tuple([int(i*255) for i in colors.to_rgb(self.targets[key].color)[::-1]])  # need BGR color ordering
            drawing.openCVCircle(img, circlePos, 15, clr, -1, subPixelFac)

        if path:
            cv2.imwrite(path, img)

        return img

