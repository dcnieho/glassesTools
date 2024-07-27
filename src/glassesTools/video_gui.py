try:
    from imgui_bundle import imgui, immapp, hello_imgui, glfw_utils, icons_fontawesome_6 as ifa6
    import glfw
    import OpenGL.GL as gl
except ImportError:
    raise ImportError('imgui_bundle (or one of its dependencies) is not installed, GUI functionality is not available')

import threading
import numpy as np
import functools
from enum import Enum, auto
import dataclasses
from typing import Any

from . import annotation, intervals, platform, timeline_gui, timestamps, utils

class Action(Enum):
    Back_Time       = auto()
    Forward_Time    = auto()
    Back_Frame      = auto()
    Forward_Frame   = auto()
    Pause           = auto()
    Quit            = auto()
    Annotate_Make   = auto()
    Annotate_Delete = auto()


shortcut_key_map = {
    Action.Back_Time        : imgui.Key.left_arrow,
    Action.Forward_Time     : imgui.Key.right_arrow,
    Action.Back_Frame       : imgui.Key.j,
    Action.Forward_Frame    : imgui.Key.k,
    Action.Pause            : imgui.Key.space,
    Action.Quit             : imgui.Key.enter,
    Action.Annotate_Delete  : imgui.Key.d
}

action_lbl_map: dict[Action, str|tuple[str,str]] = {
    Action.Back_Time        : ifa6.ICON_FA_BACKWARD_FAST,
    Action.Forward_Time     : ifa6.ICON_FA_FORWARD_FAST,
    Action.Back_Frame       : ifa6.ICON_FA_BACKWARD_STEP,
    Action.Forward_Frame    : ifa6.ICON_FA_FORWARD_STEP,
    Action.Pause            : (ifa6.ICON_FA_PLAY, ifa6.ICON_FA_PAUSE),  # (no_playing, playing)
    Action.Quit             : "Done",
    Action.Annotate_Delete  : ifa6.ICON_FA_TRASH_CAN
}

action_tooltip_map = {
    Action.Back_Time        : "Back 1 s (with shift 10 s)",
    Action.Forward_Time     : "Forward 1 s (with shift 10 s)",
    Action.Back_Frame       : "Back 1 frame (with shift 10 frames)",
    Action.Forward_Frame    : "Forward 1 frame (with shift 10 frames)",
    Action.Pause            : "Pause or resume playback",
    Action.Quit             : "Done",
    Action.Annotate_Delete  : "Delete annotation"
}

@dataclasses.dataclass
class Button:
    action: Action
    lbl: str
    tooltip: str
    key: imgui.Key
    event: annotation.Event = None
    color: imgui.ImVec4 = None

    has_shift: bool = dataclasses.field(init=False, default=False)
    repeats: bool =  dataclasses.field(init=False, default=False)
    full_tooltip: str = dataclasses.field(init=False, default='')

    def __post_init__(self):
        self.has_shift = self.action in [Action.Back_Time, Action.Back_Frame, Action.Forward_Frame, Action.Forward_Time]
        self.repeats = self.has_shift

        accelerator = imgui.get_key_name(self.key)
        if self.has_shift:
            mod_lbl = imgui.get_key_name(imgui.Key.im_gui_mod_shift)
            if mod_lbl.lower().startswith('mod'):
                mod_lbl = mod_lbl[3:]
            accelerator = f'{accelerator} or {mod_lbl}+{accelerator}'
        if self.event is not None:
            self.tooltip = f'Make {self.tooltip.lower()} annotation'
        self.full_tooltip = f'{self.tooltip} ({accelerator})'


# GUI provider for viewer and coder windows
class GUI:
    def __init__(self, use_thread=True):
        self._running = False
        self._should_exit = False
        self._use_thread = use_thread # NB: on MacOSX the GUI needs to be on the main thread, see https://github.com/pthom/hello_imgui/issues/33
        self._thread: threading.Thread = None
        self._not_shown_yet: dict[int, bool] = {}
        self._new_frame: dict[int, tuple[np.ndarray, float, int]] = {}
        self._new_frame_lock: threading.Lock = threading.Lock()
        self._current_frame: dict[int, tuple[np.ndarray, float, int]] = {}
        self._frame_size: dict[int, tuple[int,int]] = {}
        self._texID: dict[int,int] = {}

        self._duration: float = None
        self._last_frame_idx: int = None

        self._action_tooltips   = action_tooltip_map.copy()
        self._action_button_lbls= action_lbl_map.copy()
        self._shortcut_key_map  = shortcut_key_map.copy()
        self._annotate_shortcut_key_map: dict[annotation.Event, imgui.Key] = {}
        self._annotate_tooltips        : dict[annotation.Event, str]       = {}
        self._annotations_frame        : dict[annotation.Event, list[int]] = None

        self._allow_pause = False
        self._allow_seek = False
        self._allow_annotate = False
        self._allow_timeline_zoom = False
        self._timeline_show_annotation_labels = True
        self._timeline_show_info_on_hover = True
        self._is_playing = False
        self._requests: list[tuple[str,Any]] = []

        self._next_window_id: int = 0
        self._windows_lock: threading.Lock = threading.Lock()
        self._windows: dict[int,str] = {}
        self._window_flags = int(
                                    imgui.WindowFlags_.no_title_bar |
                                    imgui.WindowFlags_.no_collapse |
                                    imgui.WindowFlags_.no_scrollbar |
                                    imgui.WindowFlags_.no_resize        # no resize gripper in window's bottom-right
                                )
        self._window_visible: dict[int,bool] = {}
        self._window_determine_size: dict[int,bool] = {}
        self._window_show_controls: dict[int,bool] = {}
        self._window_show_play_percentage: dict[int,bool] = {}
        self._window_sfac: dict[int,float] = {}
        self._window_timeline: dict[int,timeline_gui.Timeline] = {}
        self._window_timecode_pos: dict[int,str] = {}

        self._buttons: dict[Action|tuple[Action,annotation.Event], Button] = {}
        self._add_remove_button(True, Action.Quit)

    def __del__(self):
        self.stop()

    def add_window(self, name: str) -> int:
        with self._windows_lock:
            w_id = self._next_window_id
            self._windows[w_id]                 = name
            self._not_shown_yet[w_id]           = True
            self._texID[w_id]                   = None
            self._new_frame[w_id]               = (None, None, -1)
            self._current_frame[w_id]           = (None, None, -1)
            self._frame_size[w_id]              = (-1, -1)
            self._window_visible[w_id]          = False
            self._window_determine_size[w_id]   = False
            self._window_show_controls[w_id]    = False
            self._window_show_play_percentage[w_id] = False
            self._window_sfac[w_id]             = 1.
            self._window_timeline[w_id]         = None
            self._window_timecode_pos[w_id]     = 'l'

            self._next_window_id += 1
            return w_id

    def delete_window(self, window_id: int):
        assert window_id!=0, 'It is not possible to delete the main window'
        with self._windows_lock:
            self._windows.pop(window_id)
            self._not_shown_yet.pop(window_id)
            self._texID.pop(window_id)
            self._new_frame.pop(window_id)
            self._current_frame.pop(window_id)
            self._frame_size.pop(window_id)
            self._window_visible.pop(window_id)
            self._window_determine_size.pop(window_id)
            self._window_show_controls.pop(window_id)
            self._window_show_play_percentage.pop(window_id)
            self._window_sfac.pop(window_id)
            self._window_timeline.pop(window_id)
            self._window_timecode_pos.pop(window_id)

    def set_allow_pause(self, allow_pause: bool):
        self._allow_pause = allow_pause
        self._add_remove_button(self._allow_pause, Action.Pause)
    def set_allow_seek(self, allow_seek: bool):
        self._allow_seek = allow_seek
        for w in self._windows:
            if self._window_timeline[w] is not None:
                self._window_timeline[w].set_allow_seek(allow_seek)
        self._add_remove_button(self._allow_seek, Action.Back_Time)
        self._add_remove_button(self._allow_seek, Action.Forward_Time)
        self._add_remove_button(self._allow_seek, Action.Back_Frame)
        self._add_remove_button(self._allow_seek, Action.Forward_Frame)
    def _add_remove_button(self, add: bool, action: Action, event: annotation.Event=None):
        d_key = (action,event) if event else action
        self._buttons.pop(d_key, None)
        if add: # NB: nothing to do for remove, already removed
            if action==Action.Annotate_Make:
                assert event is not None, f'Cannot set an annotate action without a provided event'
                lbl = event.value
                tooltip = self._annotate_tooltips[event]
                key = self._annotate_shortcut_key_map[event]
            else:
                lbl = self._action_button_lbls[action]
                tooltip = self._action_tooltips[action]
                key = self._shortcut_key_map[action]
            self._buttons[d_key] = Button(action, lbl, tooltip, key, event)

    def set_allow_timeline_zoom(self, allow_timeline_zoom: bool):
        self._allow_timeline_zoom = allow_timeline_zoom
        for w in self._windows:
            if self._window_timeline[w] is not None:
                self._window_timeline[w].set_allow_timeline_zoom(allow_timeline_zoom)
    def set_show_controls(self, show_controls: bool, window_id:int = None):
        if window_id is None:
            window_id = self._get_main_window_id()
        self._window_show_controls[window_id] = show_controls
    def set_timecode_position(self, position, window_id:int = None):
        if window_id is None:
            window_id = self._get_main_window_id()
        assert position in ['l','r'], f"For position, only 'l' and 'r' are understood, not '{position}'"
        self._window_timecode_pos[window_id] = position
    def set_show_play_percentage(self, show_play_percentage: bool, window_id:int = None):
        if window_id is None:
            window_id = self._get_main_window_id()
        self._window_show_play_percentage[window_id] = show_play_percentage
    def set_duration(self, duration: float, last_frame_idx: int):
        self._duration = duration
        self._last_frame_idx = last_frame_idx

    def set_show_timeline(self, show_timeline: bool, video_ts: timestamps.VideoTimestamps = None, annotations: dict[annotation.Event, list[int]] = None, window_id:int = None):
        if window_id is None:
            window_id = self._get_main_window_id()

        if show_timeline:
            self._window_timeline[window_id] = timeline_gui.Timeline(video_ts, annotations)
            self._window_timeline[window_id].set_allow_annotate(self._allow_annotate)
            self._window_timeline[window_id].set_allow_seek(self._allow_seek)
            self._window_timeline[window_id].set_allow_timeline_zoom(self._allow_timeline_zoom)
            self._window_timeline[window_id].set_show_annotation_labels(self._timeline_show_annotation_labels)
            self._window_timeline[window_id].set_show_info_on_hover(self._timeline_show_info_on_hover)
            self._window_timeline[window_id].set_annotation_keys(self._annotate_shortcut_key_map, self._annotate_tooltips)
            if video_ts is not None:
                last_frame_idx, duration = video_ts.get_last()
                self.set_duration(duration, last_frame_idx)
        else:
            self._window_timeline[window_id] = None
        if annotations is not None and self._any_has_timeline():
            self._annotations_frame = annotations
        self._create_annotation_buttons()
    def get_annotation_colors(self, window_id:int = None):
        if window_id is None:
            window_id = self._get_main_window_id()
        if self._window_timeline[window_id] is None:
            return None
        colors = self._window_timeline[window_id].get_annotation_colors()
        return {e:(*colors[e],) for e in colors}

    def set_allow_annotate(self, allow_annotate: bool, annotate_shortcut_key_map: dict[annotation.Event, imgui.Key]=None, annotate_tooltips: dict[annotation.Event, str] = None):
        self._allow_annotate = allow_annotate
        if annotate_shortcut_key_map is not None:
            self._annotate_shortcut_key_map = annotate_shortcut_key_map
        if annotate_tooltips is None:
            annotate_tooltips = {e:annotation.tooltip_map[e] for e in self._annotate_shortcut_key_map}
        self._annotate_tooltips = annotate_tooltips
        for w in self._windows:
            if self._window_timeline[w] is not None and self._window_timeline[w].get_num_annotations():
                self._window_timeline[w].set_allow_annotate(allow_annotate)
                self._window_timeline[w].set_annotation_keys(self._annotate_shortcut_key_map, self._annotate_tooltips)
        self._create_annotation_buttons()

    def _has_timeline(self, window_id: int):
        return self._window_timeline[window_id] is not None and self._window_timeline[window_id].get_num_annotations()
    def _any_has_timeline(self):
        for w in self._windows:
            if self._has_timeline(w):
                return True
        return False
    def _create_annotation_buttons(self):
        any_timeline = self._any_has_timeline()
        # for safety, remove all possible already registered events
        for e in annotation.Event:
            self._buttons.pop((Action.Annotate_Make, e), None)
        self._add_remove_button(any_timeline and self._allow_annotate, Action.Annotate_Delete)
        # and make buttons if we have a visible timeline
        if not any_timeline or not self._allow_annotate:
            return
        for e in self._annotate_shortcut_key_map:
            self._add_remove_button(self._allow_annotate, Action.Annotate_Make, e)

    def set_show_annotation_label(self, show_label: bool, window_id:int = None):
        self._timeline_show_annotation_labels = show_label
        if window_id is None:
            window_id = self._get_main_window_id()
        if self._window_timeline[window_id] is not None:
            self._window_timeline[window_id].set_show_annotation_labels(show_label)

    def set_show_annotation_info_on_hover(self, show_info: bool, window_id:int = None):
        self._timeline_show_info_on_hover = show_info
        if window_id is None:
            window_id = self._get_main_window_id()
        if self._window_timeline[window_id] is not None:
            self._window_timeline[window_id].set_show_info_on_hover(show_info)

    def set_playing(self, is_playing: bool):
        self._is_playing = is_playing

    def start(self):
        if not self._windows:
            raise RuntimeError('add a window (GUI.add_window) before you call start')
        if self._use_thread:
            if self._thread is not None:
                raise RuntimeError('The gui is already running, cannot start again')
            self._thread = threading.Thread(target=self._thread_start_fun)
            self._thread.start()
        else:
            self._thread_start_fun()

    def set_window_title(self, new_title: str, window_id:int = None):
        if window_id is None:
            window_id = self._get_main_window_id()
        if self._running and window_id==0:  # main window
            # this is just for show, doesn't trigger an update. But lets keep them in sync
            hello_imgui.get_runner_params().app_window_params.window_title = new_title
            # actually update window title
            win = glfw_utils.glfw_window_hello_imgui()
            glfw.set_window_title(win, new_title)
        else:
            self._windows[window_id] = new_title

    def get_requests(self):
        reqs = self._requests
        self._requests = []
        return reqs

    def stop(self):
        self._should_exit = True
        if self._thread is not None:
            self._thread.join()
        self._thread = None

    def set_frame_size(self, frame_size: tuple[int,int], window_id:int = None):
        if window_id is None:
            window_id = self._get_main_window_id()

        self._window_determine_size[window_id] = any([x!=y for x,y in zip(self._frame_size[window_id],frame_size)])
        self._frame_size[window_id] = frame_size

    def update_image(self, frame: np.ndarray, pts: float, frame_nr: int, window_id:int = None):
        # since this has an independently running loop,
        # need to update image whenever new one available
        if window_id is None:
            window_id = self._get_main_window_id()
        with self._new_frame_lock:
            self._new_frame[window_id] = (frame, pts, frame_nr) # just copy ref to frame is enough

    def notify_annotations_changed(self):
        for w in self._windows:
            if self._window_timeline[w] is not None:
                self._window_timeline[w].notify_annotations_changed()

    def _get_main_window_id(self):
        # NB: actually the main window id is always 0, but i want to have a defensive check here
        # so that user doesn't make a mistake not specifying the window they apply an operation to
        with self._windows_lock:
            if len(self._windows)==1:
                return next(iter(self._windows))
            else:
                raise RuntimeError("You have more than one window, you must indicate for which window you are making this call for")

    def _thread_start_fun(self):
        self._running = True
        self._lastT=0.
        self._should_exit = False

        def close_callback(window: glfw._GLFWwindow):
            self._requests.append(('exit',True))

        def post_init():
            imgui.get_io().config_viewports_no_decoration = False
            imgui.get_io().config_viewports_no_auto_merge = True

            glfw.swap_interval(0)
            self._window_visible[self._get_main_window_id()] = False
            glfw.set_window_close_callback(glfw_utils.glfw_window_hello_imgui(), close_callback)

        params = hello_imgui.RunnerParams()
        params.app_window_params.restore_previous_geometry = False
        params.ini_folder_type = hello_imgui.IniFolderType.temp_folder  # so we don't have endless ini files in the app folder, since we don't use them anyway (see previous line, restore_previous_geometry = False)
        params.app_window_params.window_title = self._windows[self._get_main_window_id()]
        params.app_window_params.hidden = True
        params.fps_idling.fps_idle = 0
        params.callbacks.default_icon_font = hello_imgui.DefaultIconFont.font_awesome6
        params.callbacks.show_gui  = self._gui_func
        params.callbacks.post_init = post_init

        # multiple window support
        params.imgui_window_params.config_windows_move_from_title_bar_only = True
        params.imgui_window_params.enable_viewports = True

        immapp.run(params)
        self._running = False

    def _gui_func(self):
        # check if we should exit
        if self._should_exit:
            # clean up
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            for w in self._windows:
                if self._texID[w]:
                    # delete
                    gl.glDeleteTextures(1, [self._texID[w]])
                    self._texID[w] = None
            # and kill
            hello_imgui.get_runner_params().app_shall_exit = True
            # nothing more to do
            return

        # upload texture if needed
        with self._windows_lock:
            for w in self._windows:
                with self._new_frame_lock:
                    if self._new_frame[w][0] is not None:
                        if self._texID[w] is None:
                            self._texID[w] = gl.glGenTextures(1)
                        # upload texture
                        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texID[w])
                        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
                        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
                        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self._new_frame[w][0].shape[1], self._new_frame[w][0].shape[0], 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, self._new_frame[w][0])

                        # if first time we're showing something (we have a new frame but not yet a current frame)
                        self._frame_size[w] = self._new_frame[w][0].shape

                    if self._new_frame[w][2]!=-1:
                        if self._not_shown_yet[w] and self._frame_size[w][0]!=-1:   # need to have a known framesize to be able to start showing the window
                            # tell window to resize
                            self._window_determine_size[w] = True
                            if w==0:
                                # and show window if needed
                                if not self._window_visible[w]:
                                    hello_imgui.get_runner_params().app_window_params.hidden = False
                            # mark window as shown
                            self._window_visible[w] = True
                            self._not_shown_yet[w] = False
                        elif self._new_frame[w][0] is not None:
                            # detect when frame changed size
                            self._window_determine_size[w] = any([x!=y for x,y in zip(self._frame_size[w],self._new_frame[w][0].shape)])

                        # keep record of what we're showing
                        self._current_frame[w]  = self._new_frame[w]
                        self._new_frame[w]      = (None, None, -1)

                        # inform timeline GUI of new frame's timestamp
                        if self._window_timeline[w] is not None:
                            self._window_timeline[w].set_position(*self._current_frame[w][1:])

            # show windows
            for w in self._windows.keys():
                if self._window_visible[w]:
                    self._draw_gui(w, w>0)

    def _draw_gui(self, w, need_begin_end):
        if self._current_frame[w][2]==-1 or self._frame_size[w]==-1:
            return

        # determine window size if needed
        dpi_fac = hello_imgui.dpi_window_size_factor()
        img_sz = np.array([self._frame_size[w][1]*dpi_fac, self._frame_size[w][0]*dpi_fac])
        tl_height = 0   # calc size of timeline
        if (tl:=self._window_timeline[w]) is not None:
            timeline_fixed_elements_height = tl.get_fixed_elements_height()
            tracks_height = 25*tl.get_num_annotations()*dpi_fac  # 25 pixels per track
            tl_height = int(timeline_fixed_elements_height+tracks_height)
        if self._window_determine_size[w]:
            win     = glfw_utils.glfw_window_hello_imgui()
            w_bounds= get_current_monitor(*glfw.get_window_pos(win))[1]
            w_bounds= adjust_bounds_for_framesize(w_bounds, glfw.get_window_frame_size(win))
            w_bounds.size = [w_bounds.size[0], w_bounds.size[1]-tl_height]  # adjust for timeline
            img_fit = w_bounds.ensure_window_fits_this_monitor(hello_imgui.ScreenBounds(size=[int(x) for x in img_sz]))
            self._window_sfac[w] = min([x/y for x,y in zip(img_fit.size,img_sz)])
            img_fit.size = [int(x*self._window_sfac[w]) for x in img_sz]
            if not need_begin_end:
                glfw.set_window_pos (win, *img_fit.position)
                glfw.set_window_size(win, img_fit.size[0], img_fit.size[1]+tl_height)
        elif not need_begin_end:
            win_sz = imgui.get_content_region_avail()
            win_sz.y -= tl_height
            self._window_sfac[w] = min([x/y for x,y in zip(win_sz,img_sz)])


        if need_begin_end:
            if self._window_determine_size[w]:
                imgui.set_next_window_pos(img_fit.position)
                imgui.set_next_window_size([x+y for x,y in zip(img_fit.size,(0, tl_height))])
            imgui.push_style_var(imgui.StyleVar_.window_padding, (0., 0.))
            opened, self._window_visible[w] = imgui.begin(self._windows[w], self._window_visible[w], self._window_flags)
            if not opened:
                imgui.end()
                return
            if not self._window_determine_size[w]:
                win_sz = imgui.get_content_region_avail()
                win_sz.y -= tl_height
                self._window_sfac[w] = min([x/y for x,y in zip(win_sz,img_sz)])
        self._window_determine_size[w] = False

        # draw image
        img_sz = (img_sz * self._window_sfac[w]).astype('int')
        img_space = imgui.get_content_region_avail().x
        img_margin = max((img_space-img_sz[0])/2,0)
        if self._current_frame[w][0] is None:
            overlay_text = 'No image'
            text_size = imgui.calc_text_size(overlay_text)
            offset = [(x-y)/2 for x,y in zip(img_sz,text_size)]
            imgui.set_cursor_pos((img_margin+offset[0],offset[1]))
            imgui.text_colored((1., 0., 0., 1.), overlay_text)
            imgui.set_cursor_pos((img_margin,0))
            imgui.dummy(img_sz)
        else:
            imgui.set_cursor_pos((img_margin,0))
            imgui.image(self._texID[w], img_sz)

        # prepare for drawing bottom status overlay
        fr_ts, fr_idx = self._current_frame[w][1:]
        if fr_ts is not None:
            overlay_text = f'{utils.format_duration(fr_ts,True)} ({fr_ts:.3f}) [{fr_idx}]'
        else:
            overlay_text = f'{fr_idx}'
        if self._window_show_play_percentage[w] and self._last_frame_idx is not None:
            overlay_text += f' ({fr_idx/self._last_frame_idx*100:.1f}%)'
        txt_sz = imgui.calc_text_size(overlay_text)
        overlay_size = txt_sz+imgui.ImVec2([x*2 for x in imgui.get_style().frame_padding])
        match self._window_timecode_pos[w]:
            case 'l':
                overlay_x_pos = img_margin
            case 'r':
                overlay_x_pos = img_margin+img_sz[0]-overlay_size.x
                if not self._window_show_controls[w]:
                    overlay_x_pos -= controls_child_size.x

        # prepare for drawing action buttons (may be invisible, still submit
        # them for shortcut routing)
        # collect buttons in right order
        buttons: list[Button|None] = []
        if self._allow_seek:
            buttons.append(self._buttons[Action.Back_Time])
            buttons.append(self._buttons[Action.Back_Frame])
        if self._allow_pause:
            buttons.append(self._buttons[Action.Pause])
        if self._allow_seek:
            buttons.append(self._buttons[Action.Forward_Frame])
            buttons.append(self._buttons[Action.Forward_Time])
        if self._allow_pause or self._allow_seek:
            buttons.extend([None, None])
        buttons.append(self._buttons[Action.Quit])
        if self._allow_annotate and self._has_timeline(w) and self._annotate_shortcut_key_map:
            buttons.extend([None, None])
            buttons.append(self._buttons[Action.Annotate_Delete])
            annotation_colors = self._window_timeline[w].get_annotation_colors()
            annotate_keys, annotate_ivals = intervals.which_interval(self._current_frame[w][2], self._annotations_frame)
            for e in self._annotate_shortcut_key_map:
                if e in annotation_colors and e in annotate_keys:
                    but = dataclasses.replace(self._buttons[(Action.Annotate_Make, e)])
                    but.color = annotation_colors[e]
                else:
                    but = self._buttons[(Action.Annotate_Make, e)]
                buttons.append(but)
        # determine space they take up
        def _get_text_size(b: Button|None):
            if b is None:
                return imgui.calc_text_size('')
            elif isinstance(b.lbl,tuple):
                return max([imgui.calc_text_size(l) for l in b.lbl], key=lambda x: x.x)
            return imgui.calc_text_size(b.lbl)
        text_sizes = [_get_text_size(b) for b in buttons]
        button_sizes = [imgui.ImVec2([x+2*y for x,y in zip(ts, imgui.get_style().frame_padding)]) for ts in text_sizes]
        total_button_size = functools.reduce(lambda a,b: imgui.ImVec2(a.x+b.x, max(a.y,b.y)), button_sizes)
        total_size = imgui.ImVec2(total_button_size.x+(len(buttons)-1)*imgui.get_style().item_spacing.x, total_button_size.y)
        # draw them, or info item for tooltip
        if self._window_show_controls[w]:
            buttons_x_pos = (img_space-total_size.x)/2
            # check for overlap with status overlay
            if self._window_timecode_pos[w]=='l':
                if buttons_x_pos < overlay_x_pos+overlay_size.x:
                    # try moving to just right of the status overlay
                    buttons_x_pos = overlay_x_pos+overlay_size.x+imgui.get_style().frame_padding.x
                # check we don't go off the screen
                if buttons_x_pos+total_size.x>img_space:
                    # recenter both overlay and buttons on the image, best we can do
                    overlay_button_width = overlay_size.x+imgui.get_style().frame_padding.x+total_size.x
                    overlay_x_pos = (img_space-overlay_button_width)/2
                    buttons_x_pos = overlay_x_pos+overlay_size.x+imgui.get_style().frame_padding.x
            else:
                if buttons_x_pos+total_size.x > overlay_x_pos:
                    # try moving to just left of the status overlay
                    buttons_x_pos = overlay_x_pos-total_size.x-imgui.get_style().frame_padding.x
                # check we don't go off the screen
                if buttons_x_pos < 0:
                    # recenter both overlay and buttons on the image, best we can do
                    overlay_button_width = overlay_size.x+imgui.get_style().frame_padding.x+total_size.x
                    buttons_x_pos = (img_space-overlay_button_width)/2
                    overlay_x_pos = buttons_x_pos+total_size.x+imgui.get_style().frame_padding.x
            button_cursor_pos = (buttons_x_pos,img_sz[1]-total_size.y)
            controls_child_size = total_size
        else:
            txt_sz = imgui.calc_text_size('(?)')
            button_cursor_pos = (img_margin+img_sz[0]-txt_sz.x-2*imgui.get_style().frame_padding.x, img_sz[1]-txt_sz.y-2*imgui.get_style().frame_padding.y)
            controls_child_size = txt_sz+imgui.ImVec2([x*2 for x in imgui.get_style().frame_padding])


        # draw bottom status overlay
        imgui.set_cursor_pos((overlay_x_pos,img_sz[1]-overlay_size.y))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0,0))
        imgui.push_style_color(imgui.Col_.child_bg, (0.0, 0.0, 0.0, 0.6))
        imgui.begin_child("##status_overlay", size=overlay_size)
        imgui.set_cursor_pos(imgui.get_style().frame_padding)
        imgui.text(overlay_text)
        imgui.end_child()
        imgui.pop_style_color()
        imgui.pop_style_var()


        # draw buttons
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0,0))
        imgui.push_style_color(imgui.Col_.child_bg, (0.0, 0.0, 0.0, 0.6))
        imgui.set_cursor_pos(button_cursor_pos)
        imgui.begin_child("##controls_overlay", size=controls_child_size, window_flags=imgui.WindowFlags_.no_scrollbar)
        if not self._window_show_controls[w]:
            imgui.set_cursor_pos(imgui.get_style().frame_padding)
            imgui.text('(?)')
            if imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip | imgui.HoveredFlags_.delay_normal):
                overlay_text = ''
                for b in buttons:
                    if b is not None:
                        overlay_text += f"'{imgui.get_key_name(b.key)}': {b.tooltip}\n"
                overlay_text = overlay_text[:-1]
                imgui.set_tooltip(overlay_text)

        for b,sz in zip(buttons,button_sizes):
            if b is None:
                imgui.dummy(sz)
                imgui.same_line()
                continue
            mod_key = 0
            if b.has_shift and imgui.is_key_down(imgui.Key.im_gui_mod_shift):
                # ensure the shortcut with shift is picked up
                mod_key = imgui.Key.im_gui_mod_shift
            flags = imgui.InputFlags_.route_global
            if b.repeats:
                flags |= imgui.InputFlags_.repeat
            imgui.set_next_item_shortcut(b.key|mod_key, flags=flags)
            lbl = b.lbl
            if b.action==Action.Pause:
                lbl = lbl[1] if self._is_playing else lbl[0]
            disable = False
            if b.action==Action.Annotate_Delete:
                disable = not annotate_keys # we are not in an interval
            if disable:
                imgui.begin_disabled()
            if self._window_show_controls[w]:
                if b.color is not None:
                    but_alphas = [imgui.get_style_color_vec4(b)[3] for b in [imgui.Col_.button, imgui.Col_.button_hovered, imgui.Col_.button_active]]
                    imgui.push_style_color(imgui.Col_.button,         timeline_gui.color_replace_alpha(b.color,but_alphas[0]).value)
                    imgui.push_style_color(imgui.Col_.button_hovered, timeline_gui.color_replace_alpha(timeline_gui.color_brighten(b.color, .15),but_alphas[1]).value)
                    imgui.push_style_color(imgui.Col_.button_active,  timeline_gui.color_replace_alpha(timeline_gui.color_brighten(b.color, .9 ),but_alphas[2]).value)
                    text_contrast_ratio = 1 / 1.57
                    text_color = imgui.ImColor(imgui.get_style_color_vec4(imgui.Col_.text))
                    imgui.push_style_color(imgui.Col_.text  ,         timeline_gui.color_adjust_contrast(text_color,text_contrast_ratio,b.color).value)
                activated = imgui.button(lbl, size=sz)
                if b.color is not None:
                    imgui.pop_style_color(4)
            else:
                activated = imgui.invisible_button(lbl, size=sz)
            if activated:
                match b.action:
                    case Action.Pause:
                        self._requests.append(('toggle_pause',None))
                    case Action.Back_Time:
                        self._requests.append(('delta_time', -10. if imgui.is_key_down(imgui.Key.im_gui_mod_shift) else -1.))
                    case Action.Back_Frame:
                        self._requests.append(('delta_frame', -10 if imgui.is_key_down(imgui.Key.im_gui_mod_shift) else -1))
                    case Action.Forward_Frame:
                        self._requests.append(('delta_frame',  10 if imgui.is_key_down(imgui.Key.im_gui_mod_shift) else  1))
                    case Action.Forward_Time:
                        self._requests.append(('delta_time',  10. if imgui.is_key_down(imgui.Key.im_gui_mod_shift) else  1.))
                    case Action.Quit:
                        self._requests.append(('exit',None))
                    case Action.Annotate_Make:
                        self._requests.append(('add_coding',(b.event, [self._current_frame[w][2]])))
                    case Action.Annotate_Delete:
                        for k,iv in zip(annotate_keys,annotate_ivals):
                            if len(iv)>1 and self._current_frame[w][2] in iv:
                                # on the edge of an episode, return only the edge so we don't delete the whole episode
                                self._requests.append(('delete_coding',(k,[self._current_frame[w][2]])))
                            else:
                                self._requests.append(('delete_coding',(k,iv)))
            if self._window_show_controls[w] and imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip | imgui.HoveredFlags_.delay_normal):
                imgui.set_tooltip(b.full_tooltip)
            if disable:
                imgui.end_disabled()
            imgui.same_line()
        imgui.end_child()
        imgui.pop_style_color()
        imgui.pop_style_var()

        # draw timeline, if any
        if self._window_timeline[w] is not None:
            cur_pos = imgui.get_cursor_pos()
            imgui.set_cursor_pos((cur_pos.x, cur_pos.y-imgui.get_style().item_spacing.y))
            self._window_timeline[w].draw()
            self._requests.extend(self._window_timeline[w].get_requests())

        if need_begin_end:
            imgui.pop_style_var()
            imgui.end()

def get_current_monitor(wx, wy, ww=None, wh=None):
    # so we always return something sensible
    monitor = glfw.get_primary_monitor()
    bounds  = get_monitor_work_area(monitor)
    bestoverlap = 0
    for mon in glfw.get_monitors():
        m_bounds = get_monitor_work_area(mon)

        if ww is None or wh is None:
            # check if monitor contains (wx,wy)
            if m_bounds.contains([wx, wy]):
                return mon, m_bounds
        else:
            # check overlap of window with monitor
            mx = m_bounds.position[0]
            my = m_bounds.position[0]
            mw = m_bounds.size[0]
            mh = m_bounds.size[1]
            overlap = \
                max(0, min(wx + ww, mx + mw) - max(wx, mx)) * \
                max(0, min(wy + wh, my + mh) - max(wy, my))

            if bestoverlap < overlap:
                bestoverlap = overlap
                monitor = mon
                bounds = m_bounds

    return monitor, bounds

def get_monitor_work_area(monitor):
    monitor_area = glfw.get_monitor_workarea(monitor)
    return hello_imgui.ScreenBounds(monitor_area[:2], monitor_area[2:])

def adjust_bounds_for_framesize(w_bounds, frame_size):
    if platform.os==platform.Os.Windows:
        pos = [
            w_bounds.position[0],
            w_bounds.position[1] + frame_size[1]
        ]
        size = [
            w_bounds.size[0],
            w_bounds.size[1] - frame_size[1]
        ]
    else:
        pos = [
            w_bounds.position[0] + frame_size[0],
            w_bounds.position[1] + frame_size[1],
        ]
        size = [
            w_bounds.size[0] - (frame_size[0]+frame_size[2]),
            w_bounds.size[1] - (frame_size[1]+frame_size[3])
        ]

    return hello_imgui.ScreenBounds(pos, size)