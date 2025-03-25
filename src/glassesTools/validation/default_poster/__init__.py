import pathlib
import importlib.resources
import shutil

from ... import aruco


def deploy_config(output_dir: str|pathlib.Path):
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError('the requested directory "%s" does not exist' % output_dir)

    # copy over all config files
    for r in ['markerPositions.csv', 'targetPositions.csv', 'validationSetup.txt']:
        with importlib.resources.path(__package__, r) as p:
            shutil.copyfile(p, str(output_dir/r))

    deploy_maker(output_dir)

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
    from .. import Plane
    from ..config import get_validation_setup

    output_dir = pathlib.Path(output_dir) / "all-markers"
    if not output_dir.is_dir():
        output_dir.mkdir()

    # get validation setup
    validationSetup = get_validation_setup()

    # generate and store the markers
    aruco.deploy_marker_images(output_dir, 1000, Plane.default_aruco_dict, validationSetup['markerBorderBits'])

def deploy_default_pdf(output_file_or_dir: str|pathlib.Path):
    output_file_or_dir = pathlib.Path(output_file_or_dir)
    if output_file_or_dir.is_dir():
        output_file_or_dir = output_file_or_dir / 'poster.pdf'

    with importlib.resources.path(__package__,'poster.pdf') as p:
        shutil.copyfile(p, str(output_file_or_dir))
