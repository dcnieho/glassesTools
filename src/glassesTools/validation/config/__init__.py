
from shlex import shlex
import shutil
import numpy as np
import pandas as pd
import pathlib
import importlib.resources
import typing

from ... import data_files as _data_files


def _read_glassesValidator_config_file(file):
    # read key=value pairs into dict
    lexer = shlex(file)
    lexer.whitespace += '='
    lexer.wordchars += '.[],'   # don't split extensions of filenames in the input file, and accept stuff from python list syntax
    lexer.commenters = '%'
    return dict(zip(lexer, lexer))

_default_poster_package = '.'.join(__package__.split('.')[:-1]+['default_poster'])

def get_validation_setup(config_dir: str|pathlib.Path=None, config_file: str='validationSetup.txt') -> dict[str,typing.Any]:
    if config_dir is not None:
        with (pathlib.Path(config_dir) / config_file).open() as f:
            validation_config = _read_glassesValidator_config_file(f)
    else:
        # fall back on default config included with package
        with importlib.resources.open_text(_default_poster_package, config_file) as f:
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
        return _data_files.read_coord_file(pathlib.Path(config_dir) / file)
    else:
        return _data_files.read_coord_file(file, _default_poster_package)

def get_targets(config_dir: str|pathlib.Path=None, file='targetPositions.csv') -> pd.DataFrame|None:
    return _read_coord_file(config_dir, file)

def get_markers(config_dir: str|pathlib.Path=None, file='markerPositions.csv') -> pd.DataFrame|None:
    return _read_coord_file(config_dir, file)
