# -*- coding: utf-8 -*-
# try if we have the dependencies for the GUI submodule and import it if so
try:
    import imgui_bundle
except ImportError:
    _has_GUI = False
else:
    _has_GUI = True
    from . import gui

from . import importing, validation
from .version import __version__, __url__, __author__, __email__, __description__


# ensure ffmpeg binaries needed by various submodules are on path
import ffmpeg as _ffmpeg
_ffmpeg.add_to_path()