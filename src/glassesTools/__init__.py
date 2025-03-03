# -*- coding: utf-8 -*-

from . import importing, validation
from .version import __version__, __url__, __author__, __email__, __description__


# ensure ffmpeg binaries needed by various submodules are on path
import ffmpeg as _ffmpeg
_ffmpeg.add_to_path()