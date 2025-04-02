import threading
import time
import numpy as np
from imgui_bundle import imgui, implot, immapp, hello_imgui, glfw_utils
import glfw

from .. import gaze_headref
from . import utils

class TargetPos:
    def __init__(self, timestamp: float, frame_idx: int, cam_pos: np.ndarray):
        self.timestamp = timestamp
        self.frame_idx = frame_idx
        self.cam_pos   = cam_pos

# GUI for synchronizing two signals
class GUI:
    def __init__(self, use_thread=True):
        self._should_exit = False
        self._should_init = True
        self._use_thread = use_thread # NB: on MacOSX the GUI needs to be on the main thread, see https://github.com/pthom/hello_imgui/issues/33
        self._thread = None

        self.title      = ''
        self.ival       = -1
        self._data_lock : threading.Lock = threading.Lock()
        self.gaze_data  : dict[str, np.ndarray] = {}    # ['ts', 'x', 'y']
        self.target_data: dict[str, np.ndarray] = {}
        self.offset_t   : float = 0.
        self.offset_t_ori : float = 0.
        self._offset_xy : list[float] = [0., 0.]
        self._dragging  : list[bool] = [False, False]
        self._temp_off  = np.zeros((3,))
        self._should_rescale = True
        self._running   : bool = False
        self._show_window: bool = False
        self._new_window_title: str = None
        self.is_done    : bool = None

        # cache values
        self._gaze_data_offset  : dict[str, np.ndarray] = {}    # ['ts', 'x', 'y']
        self._gaze_data_plot_pix: dict[str, np.ndarray] = {'x': None, 'y': None}
        self._last_mouse_pos    : dict[int,tuple[float,float]] = {0: None, 1: None}
        self._plot_size         : dict[int,imgui.ImVec2] = {0: None, 1: None}
        self._plot_limits       : dict[int,implot.Rect] = {0: None, 1: None}
        self._hovered           : int = None
        self._held              : int = None
        self._drag_origin       : implot.Point = None

        self._window_flags = int(
                                    imgui.WindowFlags_.no_title_bar |
                                    imgui.WindowFlags_.no_collapse |
                                    imgui.WindowFlags_.no_scrollbar |
                                    imgui.WindowFlags_.no_scroll_with_mouse
                                )

    def __del__(self):
        self.stop()

    def start(self):
        if self._use_thread:
            if self._thread is not None:
                raise RuntimeError('The gui is already running, cannot start again')
            self._thread = threading.Thread(target=self._thread_start_fun)
            self._thread.start()
        else:
            self._thread_start_fun()

    def is_running(self) -> bool:
        return self._running

    def _thread_start_fun(self):
        self._should_exit = False
        self._user_closed_window = False

        def close_callback(window: glfw._GLFWwindow):
            self._user_closed_window = True

        def post_init():
            self._running = True
            imgui.get_io().config_viewports_no_decoration = False
            imgui.get_io().config_viewports_no_auto_merge = True

            glfw.set_window_close_callback(glfw_utils.glfw_window_hello_imgui(), close_callback)

        params = hello_imgui.RunnerParams()
        params.app_window_params.restore_previous_geometry = False
        params.ini_folder_type = hello_imgui.IniFolderType.temp_folder  # so we don't have endless ini files in the app folder, since we don't use them anyway (see previous line, restore_previous_geometry = False)
        params.app_window_params.hidden = True
        params.fps_idling.fps_idle = 0
        params.callbacks.show_gui  = self._gui_func
        params.callbacks.post_init = post_init

        addons = immapp.AddOnsParams()
        addons.with_implot = True
        immapp.run(params, addons)

    def get_state(self):
        return (self._user_closed_window, self.is_done)

    def set_data(self, title: str, ival: int, gazes: dict[int,gaze_headref.Gaze], target_positions: dict[int,TargetPos],
                 offset_t = 0.):
        if not self.is_running():
            raise RuntimeError('You can only call this function once the GUI is actually running. check GUI.is_running()')
        if not gazes or not target_positions:
            raise ValueError('No gaze or target position data was provided')

        new_title = f'{title} (episode {ival})'
        self._new_window_title = new_title
        self._show_window = True
        with self._data_lock:
            self.offset_t   = offset_t
            self.offset_t_ori = offset_t
            self._offset_xy = [0., 0.]
            self.is_done    = False
            self._should_rescale = True

            self.gaze_data['ts']= np.array([s.timestamp       for fr in gazes for s in gazes[fr]],'float')/1000 # ms -> s
            self.gaze_data['x'] = np.array([s.gaze_pos_vid[0] for fr in gazes for s in gazes[fr]],'float')
            self.gaze_data['y'] = np.array([s.gaze_pos_vid[1] for fr in gazes for s in gazes[fr]],'float')

            self.target_data['ts']  = np.array([target_positions[fr].timestamp  for fr in target_positions],'float')/1000 # ms -> s
            self.target_data['x']   = np.array([target_positions[fr].cam_pos[0] for fr in target_positions],'float')
            self.target_data['y']   = np.array([target_positions[fr].cam_pos[1] for fr in target_positions],'float')

            # set begin to 0
            t0 = min([self.gaze_data['ts'][0] , self.target_data['ts'][0] ])
            self.gaze_data  ['ts'] -= t0
            self.target_data['ts'] -= t0

            # set up cache
            self._gaze_data_offset['ts']    = self.gaze_data['ts']+self.offset_t
            self._gaze_data_offset['x']     = self.gaze_data['x'].copy()
            self._gaze_data_offset['y']     = self.gaze_data['y'].copy()

    def _gui_func(self):
        # check if we should exit
        if self._should_exit:
            # and kill
            hello_imgui.get_runner_params().app_shall_exit = True
            # nothing more to do
            return

        # set window title if wanted
        if self._new_window_title:
            # this is just for show, doesn't trigger an update. But lets keep them in sync
            hello_imgui.get_runner_params().app_window_params.window_title = self._new_window_title
            # actually update window title
            win = glfw_utils.glfw_window_hello_imgui()
            glfw.set_window_title(win, self._new_window_title)
            self._new_window_title = None

        # show window
        if self._show_window:
            hello_imgui.get_runner_params().app_window_params.hidden = False

        # check we have data to plot
        if not self.gaze_data:
            return

        # check we should do some initializing (do this now, can't be done in post_init during the first frame)
        if self._should_init:
            win     = glfw_utils.glfw_window_hello_imgui()
            w_bounds= utils.get_current_monitor(*glfw.get_window_pos(win))[1]
            w_bounds= utils.adjust_bounds_for_framesize(w_bounds, glfw.get_window_frame_size(win))
            glfw.set_window_pos (win, *w_bounds.position)
            glfw.set_window_size(win, *w_bounds.size)
            self._should_init = False

        if disabled:=self.offset_t==0.:
            imgui.begin_disabled()
        if imgui.button('Reset to 0 ms'):
            self.offset_t = 0.
            self._update_cache(None)
        if disabled:
            imgui.end_disabled()
        imgui.same_line()
        if self.offset_t_ori!=0.:
            if disabled:=self.offset_t==self.offset_t_ori:
                imgui.begin_disabled()
            if imgui.button(f'Reset to {self.offset_t_ori*1000:.1f} ms'):
                self.offset_t = self.offset_t_ori
                self._update_cache(None)
            if disabled:
                imgui.end_disabled()
            imgui.same_line()
        if imgui.button('Done', (-1,0)):
            self.is_done = True

        with self._data_lock:
            # handle gaze moving with arrow keys
            move_dir = -1 if imgui.is_key_pressed(imgui.Key.left_arrow) else 1 if imgui.is_key_pressed(imgui.Key.right_arrow) else 0
            if move_dir:
                step = 1.   if imgui.is_key_down(imgui.Key.mod_ctrl)  else \
                       0.15 if imgui.is_key_down(imgui.Key.mod_shift) else \
                       0.01 if imgui.is_key_down(imgui.Key.mod_alt)   else \
                       0.05
                self.offset_t += move_dir*step
                self._update_cache(None)

            gt = self._gaze_data_offset['ts']
            gx = self._gaze_data_offset['x']
            gy = self._gaze_data_offset['y']
            if self._held is not None:
                gt = gt+self._temp_off[0]
                gx = gx+self._temp_off[1]
                gy = gy+self._temp_off[2]

            if implot.begin_subplots('##xy_plots',2,1,(-1,-1),implot.SubplotFlags_.link_all_x):
                for d in range(2):
                    if self._should_rescale:
                        implot.set_next_axes_to_fit()
                        if d==1:
                            self._should_rescale = False
                    if implot.begin_plot('##X',flags=implot.Flags_.no_mouse_text):
                        implot.setup_axis(implot.ImAxis_.x1, None if d==0 else 'time (s)')
                        implot.setup_axis(implot.ImAxis_.y1, 'horizontal coordinate (pix)' if d==0 else 'vertical coordinate (pix)')
                        if self._hovered==d:
                            implot.push_style_var(implot.StyleVar_.line_weight, implot.get_style().line_weight*2)
                            imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
                        implot.plot_line("gaze", gt, gx if d==0 else gy)
                        if self._hovered==d:
                            implot.pop_style_var()
                        implot.plot_line("target", self.target_data['ts'], self.target_data['x' if d==0 else 'y'])
                        if implot.is_plot_hovered():
                            self._do_data_drag(d)
                        if d==0:
                            # position annotation outside axes, clamping will put it in the corner
                            ax_lims = implot.get_plot_limits()
                            implot.annotation(ax_lims.x.max,ax_lims.y.min,implot.get_style().color_(implot.Col_.inlay_text),(0,0),True,f'offset: {(self.offset_t+self._temp_off[0])*1000:.1f}ms')
                        implot.end_plot()

                implot.end_subplots()

                ws = imgui.get_window_size()
                ts = imgui.calc_text_size('(?)')
                fp = imgui.get_style().frame_padding
                imgui.set_cursor_pos((ws.x-ts.x-fp.x, ws.y+imgui.get_scroll_y()-ts.y-fp.y))
                imgui.text('(?)')
                if imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    imgui.push_text_wrap_pos(min(imgui.get_font_size() * 35, ws.x))
                    imgui.text_unformatted(
                        'Drag the gaze signal to align the two signals in time with each other. '
                        'You can also shift the gaze signal in time by using the arrow keys (hold shift '
                        'for larger steps, control for extra large steps and alt for smaller). '
                        'The horizontal offset is the applied time shift (indicated by the value in the lower-right '
                        'corner of the upper plot). Any vertical shift is not stored, but can be useful when aligning the two signals. '
                        'When done aligning the two signals, press done atop the window.'
                        )
                    imgui.pop_text_wrap_pos()
                    imgui.end_tooltip()

    def _do_data_drag(self, ax_idx: int):
        ax = 'x' if ax_idx==0 else 'y'
        id_string = f"##gaze_data_drag_{ax}"
        gid = imgui.get_id(id_string)
        imgui.push_id(id_string)
        p = implot.get_plot_mouse_pos()
        plot_size = implot.get_plot_size()
        plot_limits = implot.get_plot_limits()
        invalidate1 = invalidate2 = False
        if self._last_mouse_pos[ax_idx] is None or self._last_mouse_pos[ax_idx].x!=p.x or \
           (invalidate1:=(self._plot_size[ax_idx] is None or self._plot_size[ax_idx]!=plot_size)) or \
           (invalidate2:=(self._plot_limits[ax_idx] is None or self._plot_limits[ax_idx]!=plot_limits)):
            self._last_mouse_pos[ax_idx] = p
            self._plot_size[ax_idx] = plot_size
            self._plot_limits[ax_idx] = plot_limits
            if invalidate1 or invalidate2:
                self._gaze_data_plot_pix[ax] = None
            distance = self._distance_to_gaze_signal(p, ax)
            if distance < 4.*hello_imgui.dpi_window_size_factor():
                # hovering
                self._hovered = ax_idx
            elif self._hovered==ax_idx:
                self._hovered = None
        released = False
        drag_offset = imgui.ImVec2()
        if self._hovered==ax_idx and imgui.is_mouse_clicked(imgui.MouseButton_.left):
            self._held = ax_idx
            self._drag_origin = p
            imgui.internal.set_active_id(gid,imgui.internal.get_current_window())
        elif self._held==ax_idx:
            if imgui.is_mouse_down(imgui.MouseButton_.left):
                imgui.internal.set_active_id(gid,imgui.internal.get_current_window())
                drag_offset = imgui.ImVec2(p.x-self._drag_origin.x, p.y-self._drag_origin.y)
            else:
                self._held = None
                self._drag_origin = None
                released = True

        # process drag, apply
        if self._held==ax_idx:
            self._temp_off[0]        = drag_offset.x
            self._temp_off[ax_idx+1] = drag_offset.y
        elif released:
            self.offset_t           += self._temp_off[0]
            self._offset_xy[ax_idx] += self._temp_off[ax_idx+1]
            self._temp_off          = np.zeros((3,))
            self._update_cache(ax_idx)
        imgui.pop_id()

    def _update_cache(self, ax_idx: int=None):
        self._gaze_data_offset['ts'] = self.gaze_data['ts'] + self.offset_t
        if ax_idx in [0, 2]:
            self._gaze_data_offset['x'] = self.gaze_data['x'] + self._offset_xy[0]
        elif ax_idx in [1, 2]:
            self._gaze_data_offset['y'] = self.gaze_data['y'] + self._offset_xy[1]
        self._gaze_data_plot_pix = {'x': None, 'y': None}

    def _distance_to_gaze_signal(self, pos: implot.Point, ax: str):
        if self._gaze_data_plot_pix[ax] is None:
            self._gaze_data_plot_pix[ax] = np.array([(p.x,p.y) for p in (implot.plot_to_pixels(implot.Point(x,y)) for x,y in zip(self._gaze_data_offset['ts'], self._gaze_data_offset[ax]))])

        pos = implot.plot_to_pixels(pos)
        return np.min(np.hypot(self._gaze_data_plot_pix[ax][:,0]-pos.x, self._gaze_data_plot_pix[ax][:,1]-pos.y))

    def stop(self):
        self._should_exit = True
        if self._thread is not None:
            self._thread.join()
        self._thread = None