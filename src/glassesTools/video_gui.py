try:
    from imgui_bundle import imgui, immapp, hello_imgui, glfw_utils
    import glfw
    import OpenGL.GL as gl
except ImportError:
    raise ImportError('imgui_bundle (or one of its dependencies) is not installed, GUI functionality is not available')

import time
import threading
import numpy as np


# simple GUI provider for viewer and coder windows in glassesValidator.process
class GUI:
    def __init__(self, use_thread=True):
        self._should_exit = False
        self._use_thread = use_thread # NB: on MacOSX the GUI needs to be on the main thread, see https://github.com/pthom/hello_imgui/issues/33
        self._thread = None
        self._new_frame = {}
        self._texID = {}
        self._frame_nr = {}
        self._frame_pts = {}
        self._current_frame = {}

        self._next_window_id: int = 0
        self._windows: dict[int,str] = {}
        self._window_flags = int(
                                    imgui.WindowFlags_.no_title_bar |
                                    imgui.WindowFlags_.no_collapse |
                                    imgui.WindowFlags_.no_scrollbar |
                                    imgui.WindowFlags_.no_scroll_with_mouse |
                                    imgui.WindowFlags_.always_auto_resize
                                )
        self._window_visible = {}
        self._window_frame   = {}

        self._interesting_keys = {}
        self._pressed_keys = {}

        self._draw_callback = {'main': None, 'status': None}

    def __del__(self):
        self.stop()

    def add_window(self,name: str) -> int:
        id = self._next_window_id
        self._windows[id] = name
        self._texID[id] = None
        self._new_frame[id] = (None, None, -1)
        self._current_frame[id] = (None, None, -1)
        self._window_visible[id] = False
        self._window_frame[id]   = -1

        self._next_window_id += 1
        return id

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

    def get_state(self):
        return (self._user_closed_window,)

    def stop(self):
        self._should_exit = True
        if self._thread is not None:
            self._thread.join()
        self._thread = None

    def update_image(self, frame, pts, frame_nr, window_id = None):
        # since this has an independently running loop,
        # need to update image whenever new one available
        if window_id is None:
            if len(self._windows)==1:
                window_id = self._get_main_window_id()
            else:
                raise RuntimeError("You have more than one window, you must indicate for which window you are providing an image")

        self._new_frame[window_id] = (frame, pts, frame_nr) # just copy ref to frame is enough

    def register_draw_callback(self, type: str, callback):
        # e.g. for drawing overlays
        if type not in self._draw_callback:
            raise RuntimeError('Draw callback type unknown')
        self._draw_callback[type] = callback

    def set_interesting_keys(self, keys: list[str]):
        if isinstance(keys,str):
            keys = [x for x in keys]

        self._interesting_keys = {}
        self._pressed_keys = {}
        for k in keys:
            # convert to imgui enum
            self._interesting_keys[k] = getattr(imgui.Key,k)
            # storage for presses
            self._pressed_keys[k] = [-1, False]

    def get_key_presses(self) -> dict[str,float]:
        out = {}
        thisT = time.perf_counter()
        for k in self._pressed_keys:
            if self._pressed_keys[k][0] != -1:
                t = thisT - self._pressed_keys[k][0]
                if self._pressed_keys[k][1]:
                    out[k.upper()] = t
                else:
                    out[    k    ] = t

                # clear key press so its not output again
                self._pressed_keys[k] = [-1, False]
        return out

    def _get_main_window_id(self):
        return next(iter(self._windows))

    def _thread_start_fun(self):
        self._lastT=0.
        self._should_exit = False
        self._user_closed_window = False
        self._dpi_fac = 1

        def close_callback(window: glfw._GLFWwindow):
            self._user_closed_window = True

        def post_init():
            imgui.get_io().config_viewports_no_decoration = False
            imgui.get_io().config_viewports_no_auto_merge = True

            glfw.swap_interval(0)
            self._dpi_fac = hello_imgui.dpi_window_size_factor()
            self._window_visible[self._get_main_window_id()] = False
            glfw.set_window_close_callback(glfw_utils.glfw_window_hello_imgui(), close_callback)

        params = hello_imgui.RunnerParams()
        params.app_window_params.window_geometry.size_auto = True
        params.app_window_params.restore_previous_geometry = False
        params.ini_folder_type = hello_imgui.IniFolderType.temp_folder  # so we don't have endless ini files in the app folder, since we don't use them anyway (see previous line, restore_previous_geometry = False)
        params.app_window_params.window_title = self._windows[self._get_main_window_id()]
        params.app_window_params.hidden = True
        params.fps_idling.fps_idle = 0
        params.callbacks.show_gui  = self._gui_func
        params.callbacks.post_init = post_init

        # multiple window support
        params.imgui_window_params.config_windows_move_from_title_bar_only = True
        params.imgui_window_params.enable_viewports = True

        immapp.run(params)

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

        # manual vsync with a sleep, so that other thread can run
        # thats crappy vsync, but ok for our purposes
        thisT = time.perf_counter()
        elapsedT = thisT-self._lastT
        self._lastT = thisT

        if elapsedT < 1/60:
            time.sleep(1/60-elapsedT)

        # if user wants to know about keypresses, keep record of them
        for k in self._interesting_keys:
            if imgui.is_key_pressed(self._interesting_keys[k]):
                self._pressed_keys[k] = [thisT, imgui.is_key_down(imgui.Key.im_gui_mod_shift)]

        # upload texture if needed
        for w in self._windows:
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

                # if first time we're showing something
                if self._current_frame[w][0] is None:
                    if w==0:
                        # tell window to resize
                        hello_imgui.get_runner_params().app_window_params.window_geometry.resize_app_window_at_next_frame = True
                        # and show window if needed
                        if not self._window_visible[w]:
                            hello_imgui.get_runner_params().app_window_params.hidden = False
                    # mark window as shown
                    self._window_visible[w] = True

                # keep record of what we're showing
                self._current_frame[w]  = self._new_frame[w]
                self._new_frame[w]      = (None, None, -1)

        # show windows
        for w in self._windows.keys():
            if self._window_visible[w]:
                self._draw_gui(w, w>0)

    def _draw_gui(self, w, need_begin_end):
        if self._current_frame[w] is None or self._texID[w] is None:
            return

        if need_begin_end:
            if isinstance(self._window_size[w][0], np.ndarray):
                imgui.set_next_window_size()
            opened, self._window_visible[w] = imgui.begin(self._windows[w], self._window_visible[w], self._window_flags)
            if not opened:
                imgui.end()
                return

        imgui.set_cursor_pos((0,0))
        # draw image
        self._window_frame[w] += 1
        win_sz = np.array([*imgui.get_window_size()])
        img_sz = np.array([self._current_frame[w][0].shape[1]*self._dpi_fac, self._current_frame[w][0].shape[0]*self._dpi_fac])
        if self._window_frame[w]>2: # it takes two frames for autosize to take effect
            sfac = min([np.min(win_sz/img_sz), 1.]) # clamp to 1: don't scale up
            img_sz = (img_sz * sfac).astype('int')
        imgui.image(self._texID[w], img_sz)

        # draw bottom status overlay
        txt_sz = imgui.calc_text_size('')
        win_bottom = min(self._current_frame[w][0].shape[0]*self._dpi_fac, win_sz[1]+imgui.get_scroll_y())
        imgui.set_cursor_pos_y(win_bottom-txt_sz.y)
        imgui.push_style_color(imgui.Col_.child_bg, (0.0, 0.0, 0.0, 0.6))
        imgui.begin_child("##status_overlay", size=(-imgui.FLT_MIN,txt_sz.y))
        if self._current_frame[w][1] is not None:
            imgui.text(" %8.3f [%d]" % (self._current_frame[w][1], self._current_frame[w][2]))
        else:
            imgui.text(" %d" % (self._current_frame[w][2],))
        if self._draw_callback['status']:
            self._draw_callback['status']()
        imgui.end_child()
        imgui.pop_style_color()

        if self._draw_callback['main']:
            self._draw_callback['main']()

        if need_begin_end:
            imgui.end()

def generic_tooltip_drawer(info_dict: dict[str,str]):
    ws = imgui.get_window_size()
    ts = imgui.calc_text_size('(?)')
    imgui.same_line(ws.x-ts.x)
    imgui.text('(?)')

    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(min(imgui.get_font_size() * 35, ws.x))
        text = ''
        for k in info_dict:
            text += f"'{k.upper()}': {info_dict[k]}\n"
        text = text[:-1]
        imgui.text_unformatted(text)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()

def qns_tooltip() -> dict[str,str]:
    return {
        'q': 'Quit',
        'n': 'Next',
        's': 'Screenshot'
    }