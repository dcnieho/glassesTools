import math
import numpy as np
from typing import Any
from matplotlib import ticker
from imgui_bundle import imgui, hello_imgui, icons_fontawesome_6 as ifa6

from .. import annotation, timestamps, utils

def _color_u32_replace_alpha(color: int, alpha: float) -> int:
    col = imgui.get_style_color_vec4(color)
    return imgui.color_convert_float4_to_u32((col.x, col.y, col.z, alpha))

def color_replace_alpha(color: imgui.ImColor, alpha: float) -> imgui.ImColor:
    return imgui.ImColor(color.value.x, color.value.y, color.value.z, alpha)

def _color_luminance(color: imgui.ImColor) -> float:
    return (0.2126 * color.value.x) + (0.7152 * color.value.y) + (0.0722 * color.value.z)

def _calc_contrast_ratio(a: imgui.ImColor, b: imgui.ImColor) -> float:
    y1 = _color_luminance(a)
    y2 = _color_luminance(b)
    return (y2 + 0.05) / (y1 + 0.05) if (y1 > y2) else (y1 + 0.05) / (y2 + 0.05)

def color_adjust_contrast(color: imgui.ImColor, contrast: float, bg_color: imgui.ImColor) -> imgui.ImColor:
    bg_contrast_ratio = _calc_contrast_ratio(bg_color, color)
    return imgui.ImColor(0, 0, 0, 1 - bg_contrast_ratio * 0.45) if (bg_contrast_ratio > contrast) else color

def color_brighten(color: imgui.ImColor, amount: float):
    amount = 1. / (1. + amount)
    return imgui.ImColor(1. - (1. - color.value.x) * amount, 1. - (1. - color.value.y) * amount,
                         1. - (1. - color.value.z) * amount, color.value.w)

def color_darken(color: imgui.ImColor, amount: float):
    amount = 1. / (1. + amount)
    return imgui.ImColor(color.value.x * amount, color.value.y * amount, color.value.z * amount)


class Timeline:
    def __init__(self, video_ts: timestamps.VideoTimestamps, annotations: dict[annotation.Event, list[int]] = None):
        self._video_ts = video_ts
        self._duration = self._video_ts.get_last()[1]/1000. # ms -> s

        # horizontal scroll position, in pixels
        self._h_scroll      = 0
        self._scale         = 1.
        self._new_h_scroll  = None

        self._current_time = 0.
        self._dragged_time: float = None
        self._view_scale_fac: float = None

        # axis ticker
        self._major_ticker = ticker.MaxNLocator()
        self._major_ticks_pos: np.ndarray
        self._n_minor_divs: int
        self._minor_ticks_pos: np.ndarray
        self._major_ticks_lbl: list[str]

        # information about GUI size
        self.draw_width: int = None

        # GUI interaction possibilities
        self._allow_seek = False
        self._allow_annotate = False
        self._allow_timeline_zoom = False

        # tracks
        self._annotations_frame = annotations
        self._annotations       = self._annotations_to_time()
        self._annotation_colors = self._make_annotation_colors()
        self._show_annotation_labels = True
        self._show_info_on_hover = True
        self._annotation_tooltips: dict[annotation.Event, str] = {}

        # communication with owner
        self._requests: list[tuple[str,Any]] = []


    def get_fixed_elements_height(self) -> float:
        timeline_height = imgui.get_font_size() + 2*imgui.get_style().frame_padding.y
        return timeline_height + (imgui.get_style().scrollbar_size if self._allow_timeline_zoom else 0)

    def get_num_annotations(self) -> int:
        return 0 if self._annotations_frame is None else len(self._annotations_frame)

    def set_allow_seek(self, allow_seek: bool):
        self._allow_seek = allow_seek

    def set_allow_annotate(self, allow_annotate: bool):
        self._allow_annotate = allow_annotate

    def set_allow_timeline_zoom(self, allow_timeline_zoom: bool):
        self._allow_timeline_zoom = allow_timeline_zoom

    def set_show_annotation_labels(self, show_label):
        self._show_annotation_labels = show_label

    def set_show_info_on_hover(self, show_info_on_hover: bool):
        self._show_info_on_hover = show_info_on_hover

    def set_annotation_keys(self, annotate_shortcut_key_map: dict[annotation.Event, imgui.Key], annotate_tooltips: dict[annotation.Event, str] = None):
        self._annotation_tooltips.clear()
        for e in annotation.Event:
            tool_tip = ''
            if e in annotate_tooltips:
                tool_tip = annotate_tooltips[e]
                if e in annotate_shortcut_key_map and self._allow_annotate:
                    tool_tip += f' ({imgui.get_key_name(annotate_shortcut_key_map[e])})'
            elif e in annotate_shortcut_key_map and self._allow_annotate:
                tool_tip = imgui.get_key_name(annotate_shortcut_key_map[e])

            if tool_tip:
                self._annotation_tooltips[e] = tool_tip

    def _annotations_to_time(self) -> dict[annotation.Event, list[float]]:
        if not self._annotations_frame:
            return None
        return {e:[self._video_ts.get_timestamp(i)/1000 for i in self._annotations_frame[e]] for e in self._annotations_frame}  # ms -> s

    def _make_annotation_colors(self) -> dict[annotation.Event, imgui.ImColor]:
        if not self._annotations_frame:
            return {}

        colors = utils.get_colors(len(self._annotations_frame), 0.45, 0.65)
        return {k:imgui.ImColor(*c) for k,c in zip(self._annotations_frame, colors)}

    def get_annotation_colors(self) -> dict[annotation.Event, imgui.ImColor]:
        return self._annotation_colors

    def set_position(self, time: float, frame_idx: int):
        self._current_time = max(0,min(time,self._duration))
        self._current_frame= frame_idx

    def get_requests(self) -> list[tuple[str,Any]]:
        reqs = self._requests
        self._requests = []
        return reqs

    def notify_annotations_changed(self):
        self._annotations = self._annotations_to_time()

    def _request_time(self, time: float):
        self._requests.append(('seek', time))

    def _request_delete(self, event: annotation.Event, frame_idxs: list[int]):
        if not isinstance(frame_idxs,list):
            frame_idxs = [frame_idxs]
        self._requests.append(('delete_coding', (event,frame_idxs)))

    def _calc_scale_fac(self):
        # yields s/pix on the timeline
        self._view_scale_fac = self._duration / self.draw_width

    def _pos_to_time(self, pos_x, is_window_pos) -> float:
        if is_window_pos:
            pos_x -= self.content_origin.x
        # position as fraction of self.duration
        duration_frac = pos_x * self._view_scale_fac / self._duration + self._h_scroll/self.draw_width
        # in seconds
        return duration_frac * self._duration

    def _time_to_pos(self, time: float, as_window_pos) -> float:
        pos = time / self._view_scale_fac
        if as_window_pos:
            pos += imgui.get_cursor_screen_pos().x
        return pos

    def _determine_ticks(self):
        # calculate number of major ticks
        # how big is a tick label?
        width = imgui.calc_text_size('0:00:00.000').x
        num_ticks = math.floor(self.draw_width/width/2)

        # generate ticks
        self._major_ticker.set_params(nbins=num_ticks)
        major_ticks_time = np.array([t for t in self._major_ticker.tick_values(0, self._duration) if t>=0 and t<=self._duration])

        # generate minor ticks, logic similar to matplotlib.ticker.AutoMinorLocator
        majorstep = major_ticks_time[1] - major_ticks_time[0]
        majorstep_mantissa = 10 ** (np.log10(majorstep) % 1)
        self._n_minor_divs = 5 if np.isclose(majorstep_mantissa, [1, 2.5, 5, 10]).any() else 4
        minorstep = majorstep / self._n_minor_divs
        tmax = round(self._duration / minorstep) + 1
        minor_ticks_time = (np.arange(0, tmax) * minorstep)

        # convert from time to pixels
        self._major_ticks_pos = self._time_to_pos(major_ticks_time, False)
        self._minor_ticks_pos = self._time_to_pos(minor_ticks_time, False)

        # format the values
        # determine if we have a ms part
        need_ms = (majorstep%1) > np.finfo(np.dtype(float)).eps*10
        self._major_ticks_lbl = [utils.format_duration(x, need_ms) for x in major_ticks_time]

    def _draw_time_ruler(self):
        style = imgui.get_style()
        dpi_fac = hello_imgui.dpi_window_size_factor()

        # get input state we'll need
        drag_delta = imgui.get_mouse_drag_delta(imgui.MouseButton_.left)
        mouse_pos = imgui.get_mouse_pos()
        cursor_pos = imgui.get_cursor_screen_pos()
        size = imgui.ImVec2(self.draw_width, imgui.get_font_size()+2*style.frame_padding.y)

        # set up for interaction with the timeline
        if self._allow_seek or self._allow_timeline_zoom:
            # 1. make button over whole width
            imgui.invisible_button('##play_head_control', size)
        else:
            imgui.dummy(size)

        # enable moving play head
        if self._allow_seek:
            # 2. if pressed on the button, move play head
            if imgui.is_item_activated() or (imgui.is_item_active() and abs(drag_delta.x)>.001):
                self._request_time(self._pos_to_time(mouse_pos.x, True))
                imgui.reset_mouse_drag_delta()  # mark that drag has been processed
        if self._allow_timeline_zoom:
            self._handle_zoom(imgui.is_item_hovered())

        # draw ticks
        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(cursor_pos, cursor_pos+size)

        lbl_color   = imgui.get_color_u32(imgui.Col_.text)
        tick_color  = imgui.get_color_u32(imgui.Col_.separator)

        tick_pos_y = cursor_pos.y + size.y
        for lbl,pos in zip(self._major_ticks_lbl,self._major_ticks_pos):
            rounded_gridline_pos_x = round(cursor_pos.x + pos)
            draw_list.add_text((rounded_gridline_pos_x + 4*dpi_fac, cursor_pos.y + 2*style.frame_padding.y - 2*dpi_fac), lbl_color, lbl)
            draw_list.add_line((rounded_gridline_pos_x, tick_pos_y - 10*dpi_fac), (rounded_gridline_pos_x, tick_pos_y - 2*dpi_fac), tick_color, thickness=dpi_fac)

        # draw play head
        playhead_screen_position = self._time_to_pos(self._dragged_time if self._dragged_time is not None else self._current_time, True)
        if self.get_num_annotations():
            playhead_screen_position = playhead_screen_position - size.y * 0.5
            draw_list.add_triangle_filled(
                (playhead_screen_position           , cursor_pos.y + 2.5*dpi_fac),
                (playhead_screen_position + size.y  , cursor_pos.y + 2.5*dpi_fac),
                (playhead_screen_position + size.y/2, cursor_pos.y + size.y - 2.5*dpi_fac),
                0xE553A3F9)
        else:
            playhead_line_pos = imgui.ImVec2(playhead_screen_position, cursor_pos.y)
            draw_list.add_line(playhead_line_pos,
                               playhead_line_pos + imgui.ImVec2(0, size.y),
                               0xE553A3F9,
                               thickness=2*dpi_fac)

        # draw vertical separator
        curr_pos = imgui.get_cursor_screen_pos()
        draw_list.add_line(
            (curr_pos.x, curr_pos.y - 1*dpi_fac),
            (curr_pos.x + self.draw_width, curr_pos.y - 1*dpi_fac),
            imgui.get_color_u32(imgui.Col_.separator))

        draw_list.pop_clip_rect()

    def _handle_zoom(self, is_hovered: bool):
        mouse_pos = imgui.get_mouse_pos()
        mouse_wheel = imgui.get_io().mouse_wheel
        if is_hovered and mouse_wheel!=0.:
            # clamp scaling, scale should not get smaller than 1
            delta_scale = .25*mouse_wheel
            delta_scale = max(1., self._scale+delta_scale)-self._scale
            if np.isclose(0., delta_scale):
                return

            self._scale += delta_scale
            new_h_scroll = min(max(0., self._h_scroll + mouse_pos.x*delta_scale), self.draw_width)
            if not np.isclose(self._h_scroll, new_h_scroll):
                self._new_h_scroll = new_h_scroll

    def _timepoint_interaction_logic(self, lbl: str, x_pos: float, y_ext: float, hover_info: tuple[str,float,int]=None, draggable=False, has_context_menu=True, add_episode_action=False) -> tuple[float, bool, str|None]:
        do_move = False
        action = None
        dragging = False

        if self._allow_seek or self._allow_annotate or has_context_menu or hover_info:
            dpi_fac = hello_imgui.dpi_window_size_factor()
            cursor_pos = imgui.get_cursor_screen_pos()
            imgui.set_cursor_screen_pos((x_pos-2*dpi_fac, cursor_pos.y))
            size = imgui.ImVec2(4*dpi_fac, y_ext)
            imgui.invisible_button(f'##timepoint_interactable_{lbl}', size)
            imgui.set_cursor_screen_pos(cursor_pos)

            is_active = imgui.is_item_active()
            is_deactivated = imgui.is_item_deactivated()
            is_hovered = imgui.is_item_hovered()

            if (is_active or is_hovered):
                if draggable and self._allow_seek:
                    imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
                elif self._allow_seek or self._allow_annotate:
                    imgui.set_mouse_cursor(imgui.MouseCursor_.hand)

            if hover_info and imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip | imgui.HoveredFlags_.delay_normal):
                imgui.set_tooltip(f'{hover_info[0]} {hover_info[1]}\ntimestamp: {hover_info[2]:.3f}\nframe index: {hover_info[3]}')

            drag_finished = False
            dragging = imgui.is_mouse_dragging(imgui.MouseButton_.left)
            if self._allow_seek and draggable:
                if dragging and is_active:
                    # start or update drag
                    x_pos = imgui.get_mouse_pos().x
                    self._dragged_time = self._pos_to_time(x_pos, True)
                else:
                    imgui.reset_mouse_drag_delta()  # mark that drag has been processed
                    drag_finished = self._dragged_time is not None
                    x_pos = imgui.get_mouse_pos().x
                    self._dragged_time = None
            do_move = self._allow_seek and (is_deactivated or (draggable and drag_finished))

            if has_context_menu and (self._allow_seek or self._allow_annotate) and imgui.begin_popup_context_item(f"##timepoint_context_menu_{lbl}"):
                if self._allow_seek and imgui.selectable(ifa6.ICON_FA_ARROW_RIGHT + ' go to timepoint', False)[0]:
                    do_move = True
                if self._allow_annotate and imgui.selectable(ifa6.ICON_FA_TRASH_CAN + ' remove timepoint', False)[0]:
                    action = 'delete_timepoint'
                if self._allow_annotate and add_episode_action and imgui.selectable(ifa6.ICON_FA_TRASH_CAN + ' remove episode', False)[0]:
                    action = 'delete_episode'
                imgui.end_popup()

        return x_pos, do_move, action

    def _episode_interaction_logic(self, lbl: str, x_start: float, x_end: float, y_ext: float, hover_info: tuple[str,int,list[float],list[int]]=None) -> tuple[float, bool, str|None]:
        x_pos = x_start
        do_move = False
        action = None

        if self._allow_seek or self._allow_annotate or hover_info:
            dpi_fac = hello_imgui.dpi_window_size_factor()
            cursor_pos = imgui.get_cursor_screen_pos()
            imgui.set_cursor_screen_pos((x_start+2*dpi_fac, cursor_pos.y))
            size = imgui.ImVec2(x_end-x_start-4*dpi_fac, y_ext)
            imgui.invisible_button(f'##episode_interactable_{lbl}', size)
            imgui.set_cursor_screen_pos(cursor_pos)

            if hover_info and imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip | imgui.HoveredFlags_.delay_normal):
                imgui.set_tooltip(f'{hover_info[0]} {hover_info[1]}\ntimestamps: {hover_info[2][0]:.3f} - {hover_info[2][1]:.3f}\nframe indices: {hover_info[3][0]} - {hover_info[3][1]}')

            if (self._allow_seek or self._allow_annotate) and imgui.begin_popup_context_item(f"##episode_context_menu_{lbl}"):
                if self._allow_seek and imgui.selectable(ifa6.ICON_FA_ARROW_LEFT + ' go to start', False)[0]:
                    do_move = True
                if self._allow_seek and imgui.selectable(ifa6.ICON_FA_ARROW_RIGHT + ' go to end', False)[0]:
                    x_pos = x_end
                    do_move = True
                if self._allow_annotate and imgui.selectable(ifa6.ICON_FA_TRASH_CAN + ' remove episode', False)[0]:
                    action = 'delete_episode'
                imgui.end_popup()

        return x_pos, do_move, action

    def _draw_tracks(self):
        dpi_fac = hello_imgui.dpi_window_size_factor()
        cursor_pos = imgui.get_cursor_screen_pos()
        size       = imgui.get_content_region_avail()
        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(cursor_pos, cursor_pos+size)

        # draw background, alternate color every division
        section_color = _color_u32_replace_alpha(imgui.Col_.separator, 0.12)
        for i in range(1,len(self._major_ticks_pos),2):
            start = self._major_ticks_pos[i]
            end   = self._major_ticks_pos[i+1] if i+1<len(self._major_ticks_pos) else self.draw_width
            draw_list.add_rect_filled(
                (cursor_pos.x+start, cursor_pos.y),
                (cursor_pos.x+end  , cursor_pos.y+size.y),
                section_color
            )

        # draw gridlines
        grid_color_major = _color_u32_replace_alpha(imgui.Col_.separator, 0.85)
        grid_color_minor = _color_u32_replace_alpha(imgui.Col_.separator, 0.3)
        for i,pos in enumerate(self._minor_ticks_pos):
            rounded_gridline_pos_x = round(cursor_pos.x + pos)
            if i%self._n_minor_divs==0:
                grid_color = grid_color_major
            else:
                grid_color = grid_color_minor
            draw_list.add_line((rounded_gridline_pos_x, cursor_pos.y), (rounded_gridline_pos_x, cursor_pos.y+size.y), grid_color, thickness=dpi_fac)

        # now draw the actual tracks
        height_per_track = size.y/len(self._annotations_frame)
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0,0))
        text_size = max([imgui.calc_text_size(e.value) for e in self._annotations_frame.keys()], key=lambda x: x.x)
        for e in self._annotations_frame:
            self._draw_track(e, self._annotations[e], self._annotations_frame[e], height_per_track, self._annotation_colors[e], text_size)
        imgui.pop_style_var()

        # draw player bar
        imgui.set_cursor_screen_pos(cursor_pos)
        playhead_screen_position = self._time_to_pos(self._dragged_time if self._dragged_time is not None else self._current_time, True)
        playhead_line_pos = imgui.ImVec2(playhead_screen_position, cursor_pos.y)
        draw_list.add_line(playhead_line_pos,
                           playhead_line_pos + imgui.ImVec2(0, size.y),
                           0xE553A3F9,
                           thickness=2*dpi_fac)
        new_pos, should_process, _ = self._timepoint_interaction_logic('player_bar', playhead_screen_position, size.y, None, draggable=True, has_context_menu=False)
        if should_process:
            self._request_time(self._pos_to_time(new_pos, True))

        # draw invisible button for interaction logic
        imgui.set_cursor_screen_pos(cursor_pos)
        imgui.invisible_button('##all_tracks', size)
        imgui.same_line(0,0)
        if self._allow_timeline_zoom:
            tracks_hovered = imgui.is_item_hovered()
            self._handle_zoom(tracks_hovered)

        draw_list.pop_clip_rect()

    def _draw_annotation_line(self, name: str, idx: int, time: float, frame_idx: int, seq_nr: int, height: float, border_color: int) -> str|None:
        dpi_fac = hello_imgui.dpi_window_size_factor()
        cursor_pos = imgui.get_cursor_screen_pos()
        draw_list = imgui.get_window_draw_list()
        screen_position = self._time_to_pos(time, True)
        draw_list.add_line((screen_position, cursor_pos.y),
                            (screen_position, cursor_pos.y+height),
                            border_color,
                            thickness=2*dpi_fac)
        hover_info = (name, seq_nr, time, frame_idx) if self._show_info_on_hover else None
        _, do_move, action = self._timepoint_interaction_logic(f'annotation_{name}_{idx}', screen_position, height, hover_info, draggable=False, has_context_menu=True)
        if do_move:
            self._request_time(self._pos_to_time(screen_position, True))
        return action

    def _draw_track(self, event: annotation.Event, annotations_time: list[float], annotations_frame: list[int], height: float, color: imgui.ImColor, text_size: imgui.ImVec2):
        dpi_fac = hello_imgui.dpi_window_size_factor()
        grid_color = _color_u32_replace_alpha(imgui.Col_.separator, 0.85)
        cursor_pos = imgui.get_cursor_screen_pos()
        draw_list = imgui.get_window_draw_list()
        size = imgui.ImVec2(imgui.get_content_region_avail().x, height)
        name = event.value
        draw_list.push_clip_rect(cursor_pos, cursor_pos+size)

        # draw separator below track
        y_pos = cursor_pos.y+size.y + 0.5*dpi_fac
        draw_list.add_line(
            (cursor_pos.x, y_pos),
            (cursor_pos.x + size.x, y_pos),
            grid_color, 2*dpi_fac)

        # setup for other drawing
        text_contrast_ratio = 1 / 1.57
        text_color = imgui.ImColor(imgui.get_style_color_vec4(imgui.Col_.text))
        text_color_adjusted = imgui.color_convert_float4_to_u32(color_adjust_contrast(text_color, text_contrast_ratio, color).value)
        bg_color     = imgui.color_convert_float4_to_u32(color_replace_alpha(color, color.value.w * 0.5).value)
        border_color = imgui.color_convert_float4_to_u32(color_replace_alpha(color, color.value.w * 0.9).value)

        text_offset = imgui.ImVec2((2*dpi_fac, 2*dpi_fac))
        clip_title_max = text_size + imgui.ImVec2([2*text_offset.x, 2*text_offset.y])

        # draw annotations
        if annotation.type_map[event]==annotation.Type.Interval:
            # intervals, draw as boxes
            action = None
            for m in range(0,len(annotations_time)-1,2):     # -1 to make sure we don't try incomplete intervals
                start_screen_position = self._time_to_pos(annotations_time[m],   True)
                end_screen_position   = self._time_to_pos(annotations_time[m+1], True)
                draw_list.add_rect_filled((start_screen_position, cursor_pos.y),
                                          (end_screen_position, cursor_pos.y+size.y),
                                          bg_color)
                draw_list.add_rect((start_screen_position, cursor_pos.y+dpi_fac/2),
                                   (end_screen_position, cursor_pos.y+size.y),
                                   border_color,
                                   thickness=dpi_fac)

                hover_info = (name, int(m/2)+1, annotations_time[m], annotations_frame[m]) if self._show_info_on_hover else None
                _, do_move, action = self._timepoint_interaction_logic(f'annotation_{name}_{m}', start_screen_position, size.y, hover_info, draggable=False, has_context_menu=True, add_episode_action=True)
                if do_move:
                    self._request_time(self._pos_to_time(start_screen_position, True))
                elif action=='delete_timepoint':
                    self._request_delete(event, annotations_frame[m])
                elif action=='delete_episode':
                    self._request_delete(event, annotations_frame[m:m+2])

                hover_info = (name, int(m/2)+1, annotations_time[m+1], annotations_frame[m+1]) if self._show_info_on_hover else None
                _, do_move, action = self._timepoint_interaction_logic(f'annotation_{name}_{m+1}', end_screen_position, size.y, hover_info, draggable=False, has_context_menu=True, add_episode_action=True)
                if do_move:
                    self._request_time(self._pos_to_time(end_screen_position, True))
                elif action=='delete_timepoint':
                    self._request_delete(event, annotations_frame[m+1])
                elif action=='delete_episode':
                    self._request_delete(event, annotations_frame[m:m+2])

                hover_info = (name, int(m/2)+1, annotations_time[m:m+2], annotations_frame[m:m+2]) if self._show_info_on_hover else None
                x_pos, do_move, action = self._episode_interaction_logic(f'annotation_{name}_{m}', start_screen_position, end_screen_position, size.y, hover_info)
                if do_move:
                    self._request_time(self._pos_to_time(x_pos, True))
                elif action=='delete_episode':
                    self._request_delete(event, annotations_frame[m:m+2])

            if len(annotations_time)%2:
                # unmatched marker left over, draw it
                action = self._draw_annotation_line(f'{name} unmatched!', len(annotations_time)-1, annotations_time[-1], annotations_frame[-1], math.ceil(len(annotations_time)/2), size.y, border_color)
                if action=='delete_timepoint':
                    self._request_delete(event, annotations_frame[-1])
        else:
            # points in time: draw as lines
            for i,(at,af) in enumerate(zip(annotations_time,annotations_frame)):
                action = self._draw_annotation_line(name, i, at, af, i+1, size.y, border_color)
                if action=='delete_timepoint':
                    self._request_delete(event, annotations_frame[event][i])

        # draw track header (name + background)
        if self._show_annotation_labels:
            draw_list.add_rect_filled(
                (cursor_pos.x+self._h_scroll, cursor_pos.y),
                (cursor_pos.x+self._h_scroll+clip_title_max.x, cursor_pos.y+clip_title_max.y),
                imgui.color_convert_float4_to_u32(color.value)
            )
            draw_list.add_text(
                (cursor_pos.x+text_offset.x+self._h_scroll, cursor_pos.y+text_offset.y),
                text_color_adjusted,
                name
            )
            if event in self._annotation_tooltips and self._annotation_tooltips[event]:
                cpos = imgui.get_cursor_pos()
                imgui.set_cursor_pos((cpos.x+self._h_scroll, cpos.y))
                imgui.invisible_button(f'##track_annotation_label_{name}',clip_title_max)
                if imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    imgui.text(self._annotation_tooltips[event])
                    imgui.end_tooltip()
                imgui.set_cursor_pos(cpos)

        draw_list.pop_clip_rect()
        imgui.dummy(size)   # occupy the space of the track so cursor position moves correctly

    def draw(self):
        self.content_origin = imgui.get_cursor_screen_pos()
        flags = imgui.WindowFlags_.no_background
        if self._allow_timeline_zoom:
            flags |= imgui.WindowFlags_.always_horizontal_scrollbar
        available = imgui.get_content_region_avail()
        draw_width = available.x*self._scale
        if draw_width != self.draw_width:
            self.draw_width = draw_width
            self._calc_scale_fac()
            self._determine_ticks()
        # deal with scroll requests: submit content size and scroll for next window to avoid glitch (one frame delay in scroll)
        imgui.set_next_window_content_size((self.draw_width,0))
        if self._new_h_scroll is not None:
            imgui.set_next_window_scroll((self._new_h_scroll, -1))
            self._new_h_scroll = None
        imgui.begin_child('##all', size=available, window_flags=flags)
        self._h_scroll = imgui.get_scroll_x()
        imgui.begin_child('##timeline',size=(self.draw_width,0),window_flags=imgui.WindowFlags_.no_background)
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0., 0.))
        self._draw_time_ruler()
        imgui.pop_style_var()

        # draw tracks on the timeline (if any)
        if self._annotations_frame:
            self._draw_tracks()
        imgui.end_child()
        imgui.end_child()