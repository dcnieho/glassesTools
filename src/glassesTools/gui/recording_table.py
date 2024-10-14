import dataclasses
import datetime
import typing
import threading
from imgui_bundle import imgui, icons_fontawesome_6 as ifa6

from . import utils as gui_utils
from .. import camera_recording, eyetracker, recording, utils



@dataclasses.dataclass
class Filter:
    fun: typing.Callable[[int|str, recording.Recording], bool]
    invert = False
    id: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.id = id(self)

class ColumnSpec(typing.NamedTuple):
    position: int
    name: str
    flags: int
    display_func:  typing.Callable[[recording.Recording|camera_recording.Recording],None]
    sort_key_func: typing.Callable[[int],typing.Any]
    header_lbl: str|None=None   # if set, different string than name is used for the column header. Works only for non-angled headers

class RecordingTable:
    def __init__(self,
                 recordings: dict[int|str, recording.Recording|camera_recording.Recording],
            recordings_lock: threading.Lock,
        selected_recordings: dict[int|str, bool]|None,
        extra_columns: list[ColumnSpec] = None,
        get_rec_fun: typing.Callable[[typing.Any], recording.Recording|camera_recording.Recording] = None,
        item_context_callback: typing.Callable[[int], bool] = None,
        empty_context_callback: typing.Callable[[],None] = None,
        item_remove_callback: typing.Callable[[int], None] = None
        ):

        self.recordings = recordings
        self.recordings_lock = recordings_lock
        self.selected_recordings = selected_recordings
        self.has_selected_recordings = self.selected_recordings is not None
        if not self.has_selected_recordings:
            # make an internal one
            self.selected_recordings = {iid:False for iid in self.recordings}

        self.item_context_callback  = item_context_callback
        self.empty_context_callback = empty_context_callback
        self.item_remove_callback   = item_remove_callback
        self.get_rec_fun            = lambda rec: rec
        if get_rec_fun is not None:
            self.get_rec_fun        = get_rec_fun
        self.dont_show_empty        = False

        self.is_drag_drop_source    = False

        self.sorted_recordings_ids: list[int] = []
        self.last_clicked_id: int = None
        self.require_sort: bool = True
        self.filters: list[Filter] = []
        self.filter_box_text: str = ""

        self._last_y = None
        self._has_scroll_x = None

        self._columns: list[ColumnSpec] = []
        self._show_hide_commands: dict[int,bool] = {}
        self.build_columns(extra_columns)

        self._eye_tracker_label_width: float = None
        self.table_flags: int = (
            imgui.TableFlags_.scroll_x |
            imgui.TableFlags_.scroll_y |
            imgui.TableFlags_.hideable |
            imgui.TableFlags_.sortable |
            imgui.TableFlags_.resizable |
            imgui.TableFlags_.sort_multi |
            imgui.TableFlags_.reorderable |
            imgui.TableFlags_.row_bg |
            imgui.TableFlags_.sizing_fixed_fit |
            imgui.TableFlags_.no_host_extend_y |
            imgui.TableFlags_.no_borders_in_body_until_resize
        )

    def build_columns(self, extra_columns: list[ColumnSpec] = None):
        col_names = ifa6.ICON_FA_EYE+" Eye Tracker", ifa6.ICON_FA_SIGNATURE+" Name", ifa6.ICON_FA_USER_TIE+" Participant", ifa6.ICON_FA_CLIPBOARD+" Project", ifa6.ICON_FA_STOPWATCH+" Duration", ifa6.ICON_FA_CLOCK+" Recording Start", ifa6.ICON_FA_FOLDER+" Working Directory", ifa6.ICON_FA_FOLDER+" Source Directory", ifa6.ICON_FA_TAGS+" Firmware Version", ifa6.ICON_FA_BARCODE+" Glasses Serial", ifa6.ICON_FA_BARCODE+" Recording Unit Serial", ifa6.ICON_FA_TAGS+" Recording Software Version", ifa6.ICON_FA_BARCODE+" Scene Camera Serial", ifa6.ICON_FA_CAMERA+" Video File"
        if self.has_selected_recordings:
            col_names = (ifa6.ICON_FA_SQUARE_CHECK+" Selector",)+col_names
        i_def_col = 0
        i_col = 0
        extra_columns_pos = [x.position for x in extra_columns] if extra_columns else []
        has_user_default_sort = False if extra_columns is None else any(((e.flags & imgui.TableColumnFlags_.default_sort) for e in extra_columns))
        self._columns = []
        while True:
            if i_col in extra_columns_pos:
                self._columns.append(extra_columns[extra_columns_pos.index(i_col)])
            else:
                if i_def_col>=len(col_names):
                    break
                col = self._get_column(col_names[i_def_col], i_col)
                if has_user_default_sort:
                    # ensure default sort is not set
                    col = col._replace(flags=col.flags & ~imgui.TableColumnFlags_.default_sort)
                self._columns.append(col)
                i_def_col += 1
            i_col += 1
        self._show_hide_commands.clear()

    def show_hide_columns(self, show_hide: dict[str, bool]):
        for c in show_hide:
            matching = [c in col.name for col in self._columns]
            n_matching = sum(matching)
            if n_matching==0:
                raise ValueError(f'No column found whose name contains "{c}"')
            elif n_matching>1:
                raise ValueError(f'More than one column found whose name contains "{c}": {[col.name for i,col in enumerate(self._columns) if matching[i]]}')
            c_idx = matching.index(True)
            self._show_hide_commands[c_idx] = show_hide[c]

    def _get_column(self, name: str, position: int):
        flags = imgui.TableColumnFlags_.default_hide | imgui.TableColumnFlags_.no_resize    # most columns use this one
        match name[2:]:
            case "Selector":
                flags = imgui.TableColumnFlags_.no_hide | imgui.TableColumnFlags_.no_sort | imgui.TableColumnFlags_.no_resize | imgui.TableColumnFlags_.no_reorder
                display_func = None
                sort_key_func= None
            case "Eye Tracker":
                flags = imgui.TableColumnFlags_.no_resize
                display_func = lambda rec: self.draw_eye_tracker_widget(self.get_rec_fun(rec), align=True)
                sort_key_func= lambda iid: (r.eye_tracker.value if isinstance(r:=self.get_rec_fun(self.recordings[iid]), recording.Recording) else 'Camera').lower()
            case "Name":
                flags = imgui.TableColumnFlags_.default_sort | imgui.TableColumnFlags_.no_hide | imgui.TableColumnFlags_.no_resize
                display_func = None # special case
                sort_key_func= lambda iid: self.get_rec_fun(self.recordings[iid]).name.lower()
            case "Participant":
                flags = imgui.TableColumnFlags_.no_resize
                display_func = lambda rec: imgui.text(r.participant or "Unknown") if isinstance(r:=self.get_rec_fun(rec), recording.Recording) else None
                sort_key_func= lambda iid: (r.participant if isinstance(r:=self.get_rec_fun(self.recordings[iid]), recording.Recording) else 'zzzzz').lower()
            case "Project":
                display_func = lambda rec: imgui.text(r.project or "Unknown") if isinstance(r:=self.get_rec_fun(rec), recording.Recording) else None
                sort_key_func= lambda iid: (r.project if isinstance(r:=self.get_rec_fun(self.recordings[iid]), recording.Recording) else 'zzzzz').lower()
            case "Duration":
                flags = imgui.TableColumnFlags_.no_resize
                display_func = lambda rec: imgui.text("Unknown" if (d:=self.get_rec_fun(rec).duration) is None else str(datetime.timedelta(seconds=d//1000)))
                sort_key_func= lambda iid: 0 if (d:=self.get_rec_fun(self.recordings[iid]).duration) is None else d
            case "Recording Start":
                display_func = lambda rec: imgui.text(r.start_time.display or "Unknown") if isinstance(r:=self.get_rec_fun(rec), recording.Recording) else None
                sort_key_func= lambda iid: r.start_time.value if isinstance(r:=self.get_rec_fun(self.recordings[iid]), recording.Recording) else 0
            case "Working Directory":
                display_func = lambda rec: self.draw_working_directory(self.get_rec_fun(rec))
                sort_key_func= lambda iid: self.get_rec_fun(self.recordings[iid]).working_directory.name.lower()
            case "Source Directory":
                display_func = lambda rec: self.draw_source_directory(self.get_rec_fun(rec))
                sort_key_func= lambda iid: str(self.get_rec_fun(self.recordings[iid]).source_directory).lower()
            case "Firmware Version":
                display_func = lambda rec: imgui.text(r.firmware_version or "Unknown") if isinstance(r:=self.get_rec_fun(rec), recording.Recording) else None
                sort_key_func= lambda iid: (r.firmware_version if isinstance(r:=self.get_rec_fun(self.recordings[iid]), recording.Recording) else 'zzzzz').lower()
            case "Glasses Serial":
                display_func = lambda rec: imgui.text(r.glasses_serial or "Unknown") if isinstance(r:=self.get_rec_fun(rec), recording.Recording) else None
                sort_key_func= lambda iid: (r.glasses_serial if isinstance(r:=self.get_rec_fun(self.recordings[iid]), recording.Recording) else 'zzzzz').lower()
            case "Recording Unit Serial":
                display_func = lambda rec: imgui.text(r.recording_unit_serial or "Unknown") if isinstance(r:=self.get_rec_fun(rec), recording.Recording) else None
                sort_key_func= lambda iid: (r.recording_unit_serial if isinstance(r:=self.get_rec_fun(self.recordings[iid]), recording.Recording) else 'zzzzz').lower()
            case "Recording Software Version":
                display_func = lambda rec: imgui.text(r.recording_software_version or "Unknown") if isinstance(r:=self.get_rec_fun(rec), recording.Recording) else None
                sort_key_func= lambda iid: (r.recording_software_version if isinstance(r:=self.get_rec_fun(self.recordings[iid]), recording.Recording) else 'zzzzz').lower()
            case "Scene Camera Serial":
                display_func = lambda rec: imgui.text(r.scene_camera_serial or "Unknown") if isinstance(r:=self.get_rec_fun(rec), recording.Recording) else None
                sort_key_func= lambda iid: (r.scene_camera_serial if isinstance(r:=self.get_rec_fun(self.recordings[iid]), recording.Recording) else 'zzzzz').lower()
            case "Video File":
                display_func = lambda rec: imgui.text((r.scene_video_file if isinstance(r:=self.get_rec_fun(rec), recording.Recording) else r.video_file) or "Unknown")
                sort_key_func= lambda iid: (r.scene_video_file if isinstance(r:=self.get_rec_fun(self.recordings[iid]), recording.Recording) else r.video_file).lower()
            case _:
                raise NotImplementedError()

        return ColumnSpec(position, name, flags, display_func, sort_key_func, name[2:])

    def set_act_as_drag_drop_source(self, enabled: bool):
        self.is_drag_drop_source = enabled

    def add_filter(self, filt: Filter):
        self.filters.append(filt)
        self.require_sort = True

    def remove_filter(self, iid):
        for i, search in enumerate(self.filters):
            if search.id == iid:
                self.filters.pop(i)
        self.require_sort = True

    def font_changed(self):
        self._eye_tracker_label_width = None

    def set_local_item_remover(self):
        self.item_remove_callback = self.remove_recording

    def draw(self, accent_color: tuple[float] = None, bg_color: tuple[float] = None, style_color_recording_name: bool = False, limit_outer_size: bool = False):
        if self.dont_show_empty:
            with self.recordings_lock:
                if not self.recordings:
                    imgui.text_wrapped('There are no recordings')
                    return
        outer_size = imgui.ImVec2(0,0)
        if limit_outer_size and self._last_y is not None:
            outer_size.y = self._last_y+imgui.get_style().item_spacing.y
            if self._has_scroll_x:
                outer_size.y += imgui.get_style().scrollbar_size
        if imgui.begin_table(
            f"##recording_list",
            columns=len(self._columns),
            flags=self.table_flags,
            outer_size=outer_size
        ):
            frame_height = imgui.get_frame_height()

            # Setup
            checkbox_width = frame_height
            has_angled_headers = False
            for c_idx in range(len(self._columns)):
                col = self._columns[c_idx]
                if c_idx==0 and self.has_selected_recordings:
                    imgui.table_setup_column(col.name, col.flags, init_width_or_weight=checkbox_width)
                else:
                    imgui.table_setup_column(col.name, col.flags)
                has_angled_headers = has_angled_headers or (col.flags & imgui.TableColumnFlags_.angled_header)

            # show/hide columns through API
            for c_idx in self._show_hide_commands:
                imgui.table_set_column_enabled(c_idx, self._show_hide_commands[c_idx])
            self._show_hide_commands.clear()

            # Sticky column headers and selector row
            n_row_freeze = 2 if has_angled_headers else 1
            n_col_freeze = 1 if self.has_selected_recordings else 0
            imgui.table_setup_scroll_freeze(n_col_freeze, n_row_freeze)

            with self.recordings_lock:
                if (rs:=set(self.recordings.keys())) != (rss:= set(self.selected_recordings.keys())) or rs!=set(self.sorted_recordings_ids):
                    self.require_sort = True
                    if (new_recs := rs-rss):
                        self.selected_recordings |= {iid:False for iid in new_recs}

                # Sorting
                sort_specs = imgui.table_get_sort_specs()
                sorted_recordings_ids_len = len(self.sorted_recordings_ids)
                self.sort_and_filter_recordings(sort_specs)
                if len(self.sorted_recordings_ids) < sorted_recordings_ids_len:
                    # we've just filtered out some recordings from view. Deselect those
                    # NB: will also be triggered when removing an item, doesn't matter
                    for iid in self.recordings:
                        if iid not in self.sorted_recordings_ids:
                            self.selected_recordings[iid] = False

                # Headers
                if has_angled_headers:
                    imgui.table_angled_headers_row()
                imgui.table_next_row(imgui.TableRowFlags_.headers)
                for c_idx in range(len(self._columns)):
                    if not imgui.table_set_column_index(c_idx):
                        continue
                    if c_idx==0 and self.has_selected_recordings:  # checkbox column: reflects whether all, some or none of visible recordings are selected, and allows selecting all or none
                        # get state
                        num_selected = sum([self.selected_recordings[iid] for iid in self.sorted_recordings_ids])
                        if num_selected==0:
                            # none selected
                            multi_selected_state = -1
                        elif num_selected==len(self.sorted_recordings_ids):
                            # all selected
                            multi_selected_state = 1
                        else:
                            # some selected
                            multi_selected_state = 0

                        if multi_selected_state==0:
                            imgui.internal.push_item_flag(imgui.internal.ItemFlags_.mixed_value, True)
                        clicked, new_state = gui_utils.my_checkbox("##header_checkbox", multi_selected_state==1, frame_size=(0,0), do_vertical_align=False)
                        if multi_selected_state==0:
                            imgui.internal.pop_item_flag()

                        if clicked:
                            utils.set_all(self.selected_recordings, new_state, subset = self.sorted_recordings_ids)
                    else:
                        column_name = self._columns[c_idx].header_lbl if self._columns[c_idx].header_lbl is not None else self._columns[c_idx].name
                        if imgui.table_get_column_flags(c_idx) & imgui.TableColumnFlags_.no_header_label:
                            column_name = '##'+column_name
                        imgui.table_header(column_name)

                # Loop rows
                override_color = accent_color is not None and bg_color is not None
                if override_color:
                    a=.4
                    style_selected_row = (*tuple(a*x+(1-a)*y for x,y in zip(accent_color[:3],bg_color[:3])), 1.)
                    a=.2
                    style_hovered_row  = (*tuple(a*x+(1-a)*y for x,y in zip(accent_color[:3],bg_color[:3])), 1.)
                any_selectable_clicked = False
                if self.sorted_recordings_ids and self.last_clicked_id not in self.sorted_recordings_ids:
                    # default to topmost if last_clicked unknown, or no longer on screen due to filter
                    self.last_clicked_id = self.sorted_recordings_ids[0]
                submitted_drag_drop = False
                for iid in self.sorted_recordings_ids:
                    imgui.table_next_row()

                    num_columns_drawn = 0
                    selectable_clicked = False
                    checkbox_clicked, checkbox_hovered, checkbox_out = False, False, False
                    remove_button_hovered = False
                    has_drawn_hitbox = False
                    should_remove_item = False
                    for c_idx in range(len(self._columns)):
                        if not (imgui.table_get_column_flags(c_idx) & imgui.TableColumnFlags_.is_enabled):
                            continue
                        imgui.table_set_column_index(c_idx)

                        # Row hitbox
                        if not has_drawn_hitbox:
                            # hitbox needs to be drawn before anything else on the row so that, together with imgui.set_item_allow_overlap(), hovering button
                            # or checkbox on the row will still be correctly detected.
                            # this is super finicky, but works. The below together with using a height of frame_height+cell_padding_y
                            # makes the table row only cell_padding_y/2 longer. The whole row is highlighted correctly
                            cell_padding_y = imgui.get_style().cell_padding.y
                            cur_pos_y = imgui.get_cursor_pos_y()
                            imgui.set_cursor_pos_y(cur_pos_y - cell_padding_y/2)
                            imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.)
                            imgui.push_style_var(imgui.StyleVar_.frame_padding    , (0.,0.))
                            imgui.push_style_var(imgui.StyleVar_.item_spacing     , (0.,cell_padding_y))
                            if override_color:
                                # make selectable completely transparent
                                imgui.push_style_color(imgui.Col_.header_active , (0., 0., 0., 0.))
                                imgui.push_style_color(imgui.Col_.header        , (0., 0., 0., 0.))
                                imgui.push_style_color(imgui.Col_.header_hovered, (0., 0., 0., 0.))
                            selectable_clicked, selectable_out = imgui.selectable(f"##{iid}_hitbox", self.selected_recordings[iid], flags=imgui.SelectableFlags_.span_all_columns|imgui.SelectableFlags_.allow_overlap|imgui.internal.SelectableFlagsPrivate_.select_on_click, size=(0,frame_height+cell_padding_y))
                            # instead override table row background color, if wanted
                            if override_color:
                                if selectable_out:
                                    imgui.table_set_bg_color(imgui.TableBgTarget_.row_bg0, imgui.color_convert_float4_to_u32(style_selected_row))
                                elif imgui.is_item_hovered():
                                    imgui.table_set_bg_color(imgui.TableBgTarget_.row_bg0, imgui.color_convert_float4_to_u32(style_hovered_row))
                                imgui.pop_style_color(3)
                            imgui.pop_style_var(3)
                            # act as drag/drop source, if wanted
                            if self.is_drag_drop_source and imgui.begin_drag_drop_source(imgui.DragDropFlags_.payload_auto_expire):
                                # Set payload to carry the id of our item (NB: must be an int)
                                if not isinstance(iid, int):
                                    dd_id = -1

                                imgui.set_drag_drop_payload_py_id("RECORDING", dd_id)
                                submitted_drag_drop = True
                                # Display preview
                                if isinstance(rec:=self.get_rec_fun(self.recordings[iid]), recording.Recording):
                                    imgui.text(rec.name)
                                else:
                                    imgui.text(str(rec.source_directory / rec.video_file))
                                imgui.end_drag_drop_source()

                            imgui.set_cursor_pos_y(cur_pos_y)   # instead of imgui.same_line(), we just need this part of its effect
                            selectable_right_clicked, selectables_edited = gui_utils.handle_item_hitbox_events(iid, self.selected_recordings, self.item_context_callback)
                            self.require_sort |= selectables_edited
                            has_drawn_hitbox = True

                        if num_columns_drawn==1:
                            # (Invisible) button because it aligns the following draw calls to center vertically
                            imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.)
                            imgui.push_style_var(imgui.StyleVar_.frame_padding    , (0.,imgui.get_style().frame_padding.y))
                            imgui.push_style_var(imgui.StyleVar_.item_spacing     , (0.,imgui.get_style().item_spacing.y))
                            imgui.push_style_color(imgui.Col_.button, (0.,0.,0.,0.))
                            imgui.button(f"##{iid}_id", size=(imgui.FLT_MIN, 0))
                            imgui.pop_style_color()
                            imgui.pop_style_var(3)

                            imgui.same_line()

                        if c_idx==0 and self.has_selected_recordings:
                            # Selector
                            checkbox_clicked, checkbox_out = gui_utils.my_checkbox(f"##{iid}_selected", self.selected_recordings[iid], frame_size=(0,0))
                            checkbox_hovered = imgui.is_item_hovered()
                        elif self._columns[c_idx].header_lbl=="Name":
                            if self.item_remove_callback:
                                if imgui.button(ifa6.ICON_FA_TRASH_CAN+f"##{iid}_remove"):
                                    should_remove_item = True
                                remove_button_hovered = imgui.is_item_hovered()
                                imgui.same_line()
                            self.draw_recording_name_text(self.get_rec_fun(self.recordings[iid]), accent_color if style_color_recording_name else None)
                        else:
                            self._columns[c_idx].display_func(self.recordings[iid])
                        num_columns_drawn+=1

                    # handle item removal
                    if should_remove_item:
                        self.item_remove_callback(iid)
                        self.require_sort = True

                    # handle selection logic
                    # NB: the part of this logic that has to do with right-clicks is in handle_recording_hitbox_events()
                    # NB: any_selectable_clicked is just for handling clicks not on any recording
                    any_selectable_clicked = any_selectable_clicked or selectable_clicked or selectable_right_clicked

                    self.last_clicked_id = gui_utils.selectable_item_logic(
                        iid, self.selected_recordings, self.last_clicked_id, self.sorted_recordings_ids,
                        selectable_clicked, selectable_out, overlayed_hovered=checkbox_hovered or remove_button_hovered,
                        overlayed_clicked=checkbox_clicked, new_overlayed_state=checkbox_out
                        )

                self._last_y = imgui.get_cursor_pos().y
                self._has_scroll_x = imgui.get_current_context().current_window.scrollbar_x
                last_cursor_y = imgui.get_cursor_screen_pos().y
                imgui.end_table()

                # handle click in table area outside header+contents:
                # deselect all, and if right click, show popup
                # check mouse is below bottom of last drawn row so that clicking on the one pixel empty space between selectables
                # does not cause everything to unselect or popup to open
                if imgui.is_item_clicked(imgui.MouseButton_.left) and not any_selectable_clicked and imgui.get_io().mouse_pos.y>last_cursor_y:  # NB: table header is not signalled by is_item_clicked(), so this works correctly
                    utils.set_all(self.selected_recordings, False)

                # show menu when right-clicking the empty space
                if self.empty_context_callback and imgui.get_io().mouse_pos.y>last_cursor_y and imgui.begin_popup_context_item("##recording_list_context",popup_flags=imgui.PopupFlags_.mouse_button_right | imgui.PopupFlags_.no_open_over_existing_popup):
                    utils.set_all(self.selected_recordings, False)  # deselect on right mouse click as well
                    self.empty_context_callback()
                    imgui.end_popup()

    def remove_recording(self, iid: int|str):
        self.recordings.pop(iid,None)
        self.selected_recordings.pop(iid,None)

    def draw_eye_tracker_widget(self, rec: recording.Recording|camera_recording.Recording, align=False):
        imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0)
        x_padding = 4
        if self._eye_tracker_label_width is None:
            self._eye_tracker_label_width = 0
            for et in list(eyetracker.EyeTracker):
                self._eye_tracker_label_width = max(self._eye_tracker_label_width, imgui.calc_text_size(et.value).x)
            self._eye_tracker_label_width += 2 * x_padding
        if align:
            imgui.begin_group()
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + imgui.get_style().frame_padding.y)

        # prep for drawing widget: determine its size and position and see if visible
        if isinstance(rec, recording.Recording):
            et          = rec.eye_tracker.value
            clr         = rec.eye_tracker.color
        else:
            # camera recording
            et          = 'Camera'
            clr         = (.7, .7, .7, 1.)
        iid         = imgui.get_id(et)
        label_size  = imgui.calc_text_size(et)
        size        = imgui.ImVec2(self._eye_tracker_label_width, label_size.y)
        pos         = imgui.get_cursor_screen_pos()
        bb          = imgui.internal.ImRect(pos, (pos.x+size.x, pos.y+size.y))
        imgui.internal.item_size(size, 0)
        # if visible
        if imgui.internal.item_add(bb, iid):
            # draw frame
            imgui.internal.render_frame(bb.min, bb.max, imgui.color_convert_float4_to_u32(clr), True, imgui.get_style().frame_rounding)
            # draw text on top
            imgui.push_style_color(imgui.Col_.text, (1., 1., 1., 1.))
            imgui.internal.render_text_clipped((bb.min.x+x_padding, bb.min.y), (bb.max.x-x_padding, bb.max.y), et, None, label_size, imgui.get_style().button_text_align, bb)
            imgui.pop_style_color()

        if align:
            imgui.end_group()
        imgui.pop_style_var()

    def draw_recording_name_text(self, rec: recording.Recording|camera_recording.Recording, accent_color: tuple[float] = None):
        if accent_color is not None:
            imgui.text_colored(accent_color, rec.name)
        else:
            imgui.text(rec.name)

    def draw_working_directory(self, rec: recording.Recording|camera_recording.Recording):
        imgui.text(rec.working_directory.name if rec.working_directory else "Unknown")
        if imgui.is_item_hovered():
            if rec.working_directory and rec.working_directory.is_dir():
                text = str(rec.working_directory)
            else:
                text = 'Working directory not created yet'
            gui_utils.draw_tooltip(text)

    def draw_source_directory(self, rec: recording.Recording|camera_recording.Recording):
        imgui.text(rec.source_directory.stem or "Unknown")
        if rec.source_directory and imgui.is_item_hovered():
            gui_utils.draw_tooltip(str(rec.source_directory))

    def sort_and_filter_recordings(self, sort_specs_in: imgui.TableSortSpecs):
        if sort_specs_in.specs_dirty or self.require_sort:
            ids = list(self.recordings.keys())
            sort_specs = [sort_specs_in.get_specs(i) for i in range(sort_specs_in.specs_count)]
            for sort_spec in reversed(sort_specs):
                key = self._columns[sort_spec.column_index].sort_key_func
                ids.sort(key=key, reverse=sort_spec.get_sort_direction()==imgui.SortDirection.descending)
            self.sorted_recordings_ids = ids
            for flt in self.filters:
                key = lambda iid: flt.invert != flt.fun(iid, self.recordings[iid])
                if key is not None:
                    self.sorted_recordings_ids = list(filter(key, self.sorted_recordings_ids))
            if self.filter_box_text:
                search = self.filter_box_text.lower()
                def key(iid):
                    rec = self.get_rec_fun(self.recordings[iid])
                    if isinstance(rec, recording.Recording):
                        return \
                            search in rec.eye_tracker.value.lower() or \
                            search in rec.name.lower() or \
                            search in rec.participant.lower() or \
                            search in rec.project.lower()
                    else:
                        return \
                            search in 'camera' or \
                            search in rec.name.lower() or \
                            search in rec.video_file.lower()
                self.sorted_recordings_ids = list(filter(key, self.sorted_recordings_ids))
            sort_specs_in.specs_dirty = False
            self.require_sort = False