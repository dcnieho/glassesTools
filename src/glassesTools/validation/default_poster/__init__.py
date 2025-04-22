import pathlib
import importlib.resources
import shutil

from ... import aruco


def deploy_config(output_dir: str|pathlib.Path, overwrite=False) -> list[str]:
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError('the requested directory "%s" does not exist' % output_dir)

    # copy over all config files
    not_copied: list[str] = []
    for r in ['markerPositions.csv', 'targetPositions.csv', 'validationSetup.txt']:
        out_file = output_dir/r
        if out_file.exists() and not overwrite:
            not_copied.append(r)
            continue
        with importlib.resources.path(__package__, r) as p:
            shutil.copyfile(p, out_file)

    not_copied.extend(deploy_maker(output_dir))
    return not_copied

def deploy_maker(output_dir: str|pathlib.Path, overwrite=False) -> list[str]:
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError('the requested directory "%s" does not exist' % output_dir)

    # copy over all files
    not_copied: list[str] = []
    for r in ['poster.tex']:
        out_file = output_dir/r
        if out_file.exists() and not overwrite:
            not_copied.append(r)
            continue
        with importlib.resources.path(__package__, r) as p:
            shutil.copyfile(p, out_file)

    deploy_marker_images(output_dir)    # N.B. these can be safely overwritten
    return not_copied

def deploy_marker_images(output_dir: str|pathlib.Path):
    from ..config import get_validation_setup

    output_dir = pathlib.Path(output_dir) / "all-markers"
    if not output_dir.is_dir():
        output_dir.mkdir()

    # get validation setup
    validationSetup = get_validation_setup()

    # generate and store the markers
    aruco.deploy_marker_images(output_dir, 1000, validationSetup['arucoDictionary'], validationSetup['markerBorderBits'])

def deploy_default_pdf(output_file_or_dir: str|pathlib.Path):
    output_file_or_dir = pathlib.Path(output_file_or_dir)
    if output_file_or_dir.is_dir():
        output_file_or_dir = output_file_or_dir / 'poster.pdf'

    with importlib.resources.path(__package__,'poster.pdf') as p:
        shutil.copyfile(p, str(output_file_or_dir))
