# all the files in this module need imgui_bundle, so catch here if its not installed
try:
    import imgui_bundle
except ImportError:
    raise ImportError('imgui_bundle (or one of its dependencies) is not installed, GUI functionality is not available. You must install glassesTools with the [GUI] extra if you wish to use the GUI.') from None

from . import file_picker, msg_box, recording_table, signal_sync, timeline, utils, video_player, worldgaze