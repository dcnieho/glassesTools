
from shlex import shlex
import shutil
import numpy as np
import pandas as pd
import pathlib
import importlib.resources
import typing

from glassesTools import data_files

from . import plane


def _read_glassesValidator_config_file(file):
    # read key=value pairs into dict
    lexer = shlex(file)
    lexer.whitespace += '='
    lexer.wordchars += '.[],'   # don't split extensions of filenames in the input file, and accept stuff from python list syntax
    lexer.commenters = '%'
    return dict(zip(lexer, lexer))

def get_validation_setup(config_dir: str|pathlib.Path=None, config_file: str='validationSetup.txt') -> dict[str,typing.Any]:
    if config_dir is not None:
        with (pathlib.Path(config_dir) / config_file).open() as f:
            validation_config = _read_glassesValidator_config_file(f)
    else:
        # fall back on default config included with package
        with importlib.resources.open_text(__package__, config_file) as f:
            validation_config = _read_glassesValidator_config_file(f)

    # parse numerics into int or float
    for key,val in validation_config.items():
        if np.all([c.isdigit() for c in val]):
            validation_config[key] = int(val)
        else:
            try:
                validation_config[key] = float(val)
            except:
                pass # just keep value as a string
    return validation_config


def _read_coord_file(config_dir: str|pathlib.Path|None, file: str) -> pd.DataFrame|None:
    if config_dir is not None:
        return data_files.read_coord_file(pathlib.Path(config_dir) / file)
    else:
        return data_files.read_coord_file(file, __package__)

def get_targets(config_dir: str|pathlib.Path=None, file='targetPositions.csv') -> pd.DataFrame|None:
    return _read_coord_file(config_dir, file)

def get_markers(config_dir: str|pathlib.Path=None, file='markerPositions.csv') -> pd.DataFrame|None:
    return _read_coord_file(config_dir, file)


def deploy_validation_config(output_dir: str|pathlib.Path):
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError('the requested directory "%s" does not exist' % output_dir)

    # copy over all config files
    for r in ['markerPositions.csv', 'targetPositions.csv', 'validationSetup.txt']:
        with importlib.resources.path(__package__, r) as p:
            shutil.copyfile(p, str(output_dir/r))

    # copy over poster tex file
    poster_dir = output_dir / 'poster'
    if not poster_dir.is_dir():
        poster_dir.mkdir()

    plane.deploy_maker(poster_dir)