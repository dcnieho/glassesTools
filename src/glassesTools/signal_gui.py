try:
    from imgui_bundle import imgui, implot, immapp, hello_imgui, glfw_utils
    import glfw
except ImportError:
    raise ImportError('imgui_bundle (or one of its dependencies) is not installed, GUI functionality is not available')

import threading
import numpy as np

from . import gaze_headref

class TargetPos:
    def __init__(self, timestamp: float, frame_idx: int, cam_pos: np.ndarray):
        self.timestamp = timestamp
        self.frame_idx = frame_idx
        self.cam_pos   = cam_pos

# GUI for synchronizing two signals
class GUI:
    def __init__(self, use_thread=True):
        self._should_exit = False
        self._use_thread = use_thread # NB: on MacOSX the GUI needs to be on the main thread, see https://github.com/pthom/hello_imgui/issues/33
        self._thread = None

        self.title      = ''
        self.ival       = -1
        self.gaze_data  : dict[str, np.ndarray] = {}    # ['ts', 'x', 'y']
        self.target_data: dict[str, np.ndarray] = {}
        self.offset_t   : float = 0.
        self._offset_xy : list[float] = [0., 0.]
        self._dragging  : list[bool] = [False, False]
        self._temp_off  = np.zeros((3,))
        self._should_rescale = True
        self.is_done    : bool = None

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

    def _thread_start_fun(self):
        self._should_exit = False
        self._user_closed_window = False

        def close_callback(window: glfw._GLFWwindow):
            self._user_closed_window = True

        def post_init():
            imgui.get_io().config_viewports_no_decoration = False
            imgui.get_io().config_viewports_no_auto_merge = True

            glfw.set_window_close_callback(glfw_utils.glfw_window_hello_imgui(), close_callback)

        params = hello_imgui.RunnerParams()
        params.app_window_params.window_geometry.full_screen_mode = hello_imgui.FullScreenMode.full_monitor_work_area
        params.app_window_params.restore_previous_geometry = False
        params.ini_folder_type = hello_imgui.IniFolderType.temp_folder  # so we don't have endless ini files in the app folder, since we don't use them anyway (see previous line, restore_previous_geometry = False)
        params.app_window_params.hidden = True
        params.fps_idling.fps_idle = 0
        params.callbacks.show_gui  = self._gui_func
        params.callbacks.post_init = post_init

        # multiple window support
        params.imgui_window_params.config_windows_move_from_title_bar_only = True
        params.imgui_window_params.enable_viewports = True

        addons = immapp.AddOnsParams()
        addons.with_implot = True
        immapp.run(params, addons)

    def get_state(self):
        return (self._user_closed_window, self.is_done)

    def set_data(self, title: str, ival: int, gazes: dict[int,gaze_headref.Gaze], target_positions: dict[int,TargetPos],
                 offset_t = 0.):
        if not gazes or not target_positions:
            raise ValueError('No gaze or target position data was provided')

        new_title = f'{title} (episode {ival})'
        # this is just for show, doesn't trigger an update. But lets keep them in sync
        hello_imgui.get_runner_params().app_window_params.window_title = new_title
        # actually update window title
        win = glfw_utils.glfw_window_hello_imgui()
        glfw.set_window_title(win, new_title)
        hello_imgui.get_runner_params().app_window_params.window_geometry.resize_app_window_at_next_frame = True
        hello_imgui.get_runner_params().app_window_params.hidden = False
        self.offset_t   = offset_t
        self._offset_xy = [0., 0.]
        self.is_done    = False
        self._should_rescale = True

        self.gaze_data['ts']= np.array([s.timestamp       for fr in gazes for s in gazes[fr]])/1000 # ms -> s
        self.gaze_data['x'] = np.array([s.gaze_pos_vid[0] for fr in gazes for s in gazes[fr]])
        self.gaze_data['y'] = np.array([s.gaze_pos_vid[1] for fr in gazes for s in gazes[fr]])

        self.target_data['ts']  = np.array([target_positions[fr].timestamp  for fr in target_positions])/1000 # ms -> s
        self.target_data['x']   = np.array([target_positions[fr].cam_pos[0] for fr in target_positions])
        self.target_data['y']   = np.array([target_positions[fr].cam_pos[1] for fr in target_positions])

        # set begin to 0
        t0 = min([self.gaze_data['ts'][0] , self.target_data['ts'][0] ])
        self.gaze_data  ['ts'] -= t0
        self.target_data['ts'] -= t0

    def _gui_func(self):
        # check if we should exit
        if self._should_exit:
            # and kill
            hello_imgui.get_runner_params().app_shall_exit = True
            # nothing more to do
            return

        # check we have data to plot
        if not self.gaze_data:
            return

        if imgui.button('Done', (-1,0)):
            self.is_done = True

        toff = self.offset_t
        xoff = self._offset_xy[0]
        yoff = self._offset_xy[1]
        if any(self._dragging):
            toff += self._temp_off[0]
            xoff += self._temp_off[1]
            yoff += self._temp_off[2]
        if implot.begin_subplots('##xy_plots',2,1,(-1,-1),implot.SubplotFlags_.link_all_x)[0]:
            if self._should_rescale:
                implot.set_next_axes_to_fit()
            if implot.begin_plot('##X',flags=implot.Flags_.no_mouse_text):
                implot.setup_axis(implot.ImAxis_.x1, None)
                implot.setup_axis(implot.ImAxis_.y1, 'horizontal coordinate (pix)')
                implot.plot_line("gaze", self.gaze_data['ts']+toff, self.gaze_data['x']+xoff)
                implot.plot_line("target", self.target_data['ts'], self.target_data['x'])
                self._do_drag(0)
                # position annotation outside axes, clamping will put it in the corner
                ax_lims = implot.get_plot_limits()
                implot.annotation(ax_lims.x.max,ax_lims.y.min,(0,0,0,0),(0,0),True,f'offset: {toff*1000:.1f}ms')
                implot.end_plot()

            if self._should_rescale:
                implot.set_next_axes_to_fit()
                self._should_rescale = False
            if implot.begin_plot('##Y',flags=implot.Flags_.no_mouse_text):
                implot.setup_axis(implot.ImAxis_.x1, 'time (s)')
                implot.setup_axis(implot.ImAxis_.y1, 'vertical coordinate (pix)')
                implot.plot_line("gaze", self.gaze_data['ts']+toff, self.gaze_data['y']+yoff)
                implot.plot_line("target", self.target_data['ts'], self.target_data['y'])
                self._do_drag(1)
                implot.end_plot()

            implot.end_subplots()

    def _do_drag(self, ax_idx):
        ax_lims = implot.get_plot_limits()
        pos = ((ax_lims.x.min+ax_lims.x.max)/2., (ax_lims.y.min+ax_lims.y.max)/2.)
        held, dx, dy = implot.drag_point(1,*pos,imgui.ImVec4(0,0.9,0,1))[:3]
        if held:
            self._dragging[ax_idx]  = True
            self._temp_off[0]       = dx-pos[0]
            self._temp_off[ax_idx+1]= dy-pos[1]
        elif self._dragging[ax_idx]:
            self.offset_t           += self._temp_off[0]
            self._offset_xy[ax_idx] += self._temp_off[ax_idx+1]
            self._temp_off          = np.zeros((3,))
            self._dragging[ax_idx]  = False

    def stop(self):
        self._should_exit = True
        if self._thread is not None:
            self._thread.join()
        self._thread = None