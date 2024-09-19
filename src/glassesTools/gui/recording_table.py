import dataclasses
import datetime
import typing
from imgui_bundle import imgui, icons_fontawesome_6 as ifa6

from . import utils as gui_utils
from .. import eyetracker, recording, utils



@dataclasses.dataclass
class Filter:
    fun: typing.Callable[[recording.Recording], bool]
    invert = False
    id: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.id = id(self)

class ColumnSpec(typing.NamedTuple):
    position: int
    name: str
    flags: int
    display_func:  typing.Callable[[recording.Recording],None]
    sort_key_func: typing.Callable[[int],typing.Any]
    header_lbl: str|None=None   # if set, different string than name is used for the column header

class RecordingTable():
    def __init__(self,
                 recordings: dict[int, recording.Recording],
        selected_recordings: dict[int, bool],
        extra_columns: list[ColumnSpec] = None,
        item_context_callback: typing.Callable[[int], bool] = None,
        empty_context_callback: typing.Callable[[],None] = None,
        item_remove_callback: typing.Callable[[int], None] = None
        ):

        self.recordings = recordings
        self.selected_recordings = selected_recordings

        self._columns: list[ColumnSpec] = []
        col_names = ifa6.ICON_FA_SQUARE_CHECK+" Selector", ifa6.ICON_FA_EYE+" Eye Tracker", ifa6.ICON_FA_SIGNATURE+" Name", ifa6.ICON_FA_USER_TIE+" Participant", ifa6.ICON_FA_CLIPBOARD+" Project", ifa6.ICON_FA_STOPWATCH+" Duration", ifa6.ICON_FA_CLOCK+" Recording Start", ifa6.ICON_FA_FOLDER+" Working Directory", ifa6.ICON_FA_FOLDER+" Source Directory", ifa6.ICON_FA_TAGS+" Firmware Version", ifa6.ICON_FA_BARCODE+" Glasses Serial", ifa6.ICON_FA_BARCODE+" Recording Unit Serial", ifa6.ICON_FA_TAGS+" Recording Software Version", ifa6.ICON_FA_BARCODE+" Scene Camera Serial"
        i_def_col = 0
        i_col = 0
        extra_columns_pos = [x.position for x in extra_columns] if extra_columns else []
        while True:
            if i_col in extra_columns_pos:
                self._columns.append(extra_columns[extra_columns_pos.index(i_col)])
            else:
                if i_def_col>=len(col_names):
                    break
                self._columns.append(self._get_column(col_names[i_def_col], i_col))
                i_def_col += 1
            i_col += 1

        self.item_context_callback: typing.Callable = item_context_callback
        self.empty_context_callback = empty_context_callback
        self.item_remove_callback: typing.Callable = item_remove_callback

        self.sorted_recordings_ids: list[int] = []
        self.last_clicked_id: int = None
        self.require_sort: bool = True
        self.filters: list[Filter] = []
        self.filter_box_text: str = ""

        self._num_recordings = len(self.recordings)
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

    def _get_column(self, name: str, position: int):
        flags = imgui.TableColumnFlags_.default_hide | imgui.TableColumnFlags_.no_resize    # most columns use this one
        match name[2:]:
            case "Selector":
                flags = imgui.TableColumnFlags_.no_hide | imgui.TableColumnFlags_.no_sort | imgui.TableColumnFlags_.no_resize | imgui.TableColumnFlags_.no_reorder
                display_func = None
                sort_key_func= None
            case "Eye Tracker":
                flags = imgui.TableColumnFlags_.no_resize
                display_func = lambda rec: self.draw_eye_tracker_widget(rec, align=True)
                sort_key_func= lambda iid: self.recordings[iid].eye_tracker.value
            case "Name":
                flags = imgui.TableColumnFlags_.default_sort | imgui.TableColumnFlags_.no_hide | imgui.TableColumnFlags_.no_resize
                display_func = None # special case
                sort_key_func= lambda iid: self.recordings[iid].name.lower()
            case "Participant":
                flags = imgui.TableColumnFlags_.no_resize
                display_func = lambda rec: imgui.text(rec.participant or "Unknown")
                sort_key_func= lambda iid: self.recordings[iid].participant.lower()
            case "Project":
                display_func = lambda rec: imgui.text(rec.project or "Unknown")
                sort_key_func= lambda iid: self.recordings[iid].project.lower()
            case "Duration":
                flags = imgui.TableColumnFlags_.no_resize
                display_func = lambda rec: imgui.text("Unknown" if (d:=rec.duration) is None else str(datetime.timedelta(seconds=d//1000)))
                sort_key_func= lambda iid: 0 if (d:=self.recordings[iid].duration) is None else d
            case "Recording Start":
                display_func = lambda rec: imgui.text(rec.start_time.display or "Unknown")
                sort_key_func= lambda iid: self.recordings[iid].start_time.value
            case "Working Directory":
                display_func = lambda rec: self.draw_working_directory(rec)
                sort_key_func= lambda iid: self.recordings[iid].working_directory.name.lower()
            case "Source Directory":
                display_func = lambda rec: self.draw_source_directory(rec)
                sort_key_func= lambda iid: str(self.recordings[iid].source_directory).lower()
            case "Firmware Version":
                display_func = lambda rec: imgui.text(rec.firmware_version or "Unknown")
                sort_key_func= lambda iid: self.recordings[iid].firmware_version.lower()
            case "Glasses Serial":
                display_func = lambda rec: imgui.text(rec.glasses_serial or "Unknown")
                sort_key_func= lambda iid: self.recordings[iid].glasses_serial.lower()
            case "Recording Unit Serial":
                display_func = lambda rec: imgui.text(rec.recording_unit_serial or "Unknown")
                sort_key_func= lambda iid: self.recordings[iid].recording_unit_serial.lower()
            case "Recording Software Version":
                display_func = lambda rec: imgui.text(rec.recording_software_version or "Unknown")
                sort_key_func= lambda iid: self.recordings[iid].recording_software_version.lower()
            case "Scene Camera Serial":
                display_func = lambda rec: imgui.text(rec.scene_camera_serial or "Unknown")
                sort_key_func= lambda iid: self.recordings[iid].scene_camera_serial.lower()
            case _:
                raise NotImplementedError()

        return ColumnSpec(position, name, flags, display_func, sort_key_func, name[2:])

    def add_filter(self, filter):
        self.filters.append(filter)
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

    def draw(self, accent_color: tuple[float] = None, bg_color: tuple[float] = None, style_color_recording_name: bool = False):
        if imgui.begin_table(
            f"##recording_list",
            columns=len(self._columns),
            flags=self.table_flags
        ):
            if (num_recordings := len(self.recordings)) != self._num_recordings:
                self._num_recordings = num_recordings
                self.require_sort = True
            frame_height = imgui.get_frame_height()

            # Setup
            checkbox_width = frame_height
            for c_idx in range(len(self._columns)):
                col = self._columns[c_idx]
                extra = {}
                if c_idx==0:
                    extra['init_width_or_weight'] = checkbox_width
                imgui.table_setup_column(col.name, col.flags, **extra)

            imgui.table_setup_scroll_freeze(1, 1)  # Sticky column headers and selector row

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
            imgui.table_next_row(imgui.TableRowFlags_.headers)
            for c_idx in range(len(self._columns)):
                imgui.table_set_column_index(c_idx)
                if c_idx==0:  # checkbox column: reflects whether all, some or none of visible recordings are selected, and allows selecting all or none
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
                    imgui.table_header(self._columns[c_idx].header_lbl)

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
            for iid in self.sorted_recordings_ids:
                imgui.table_next_row()

                recording = self.recordings[iid]
                num_columns_drawn = 0
                selectable_clicked = False
                checkbox_clicked, checkbox_hovered = False, False
                remove_button_hovered = False
                has_drawn_hitbox = False
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
                        # instead override table row background color
                        if override_color:
                            if selectable_out:
                                imgui.table_set_bg_color(imgui.TableBgTarget_.row_bg0, imgui.color_convert_float4_to_u32(style_selected_row))
                            elif imgui.is_item_hovered():
                                imgui.table_set_bg_color(imgui.TableBgTarget_.row_bg0, imgui.color_convert_float4_to_u32(style_hovered_row))
                            imgui.pop_style_color(3)
                        imgui.pop_style_var(3)
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

                    if c_idx==0:
                        # Selector
                        checkbox_clicked, checkbox_out = gui_utils.my_checkbox(f"##{iid}_selected", self.selected_recordings[iid], frame_size=(0,0))
                        checkbox_hovered = imgui.is_item_hovered()
                    elif self._columns[c_idx].header_lbl=="Name":
                        if self.item_remove_callback:
                            if imgui.button(ifa6.ICON_FA_TRASH_CAN+f"##{iid}_remove"):
                                self.item_remove_callback(iid)
                                self.require_sort = True
                            remove_button_hovered = imgui.is_item_hovered()
                            imgui.same_line()
                        self.draw_recording_name_text(recording, accent_color if style_color_recording_name else None)
                    else:
                        self._columns[c_idx].display_func(recording)
                    num_columns_drawn+=1

                # handle selection logic
                # NB: the part of this logic that has to do with right-clicks is in handle_recording_hitbox_events()
                # NB: any_selectable_clicked is just for handling clicks not on any recording
                any_selectable_clicked = any_selectable_clicked or selectable_clicked or selectable_right_clicked

                self.last_clicked_id = gui_utils.selectable_item_logic(
                    iid, self.selected_recordings, self.last_clicked_id, self.sorted_recordings_ids,
                    selectable_clicked, selectable_out, overlayed_hovered=checkbox_hovered or remove_button_hovered,
                    overlayed_clicked=checkbox_clicked, new_overlayed_state=checkbox_out
                    )

            last_y = imgui.get_cursor_screen_pos().y
            imgui.end_table()

            # handle click in table area outside header+contents:
            # deselect all, and if right click, show popup
            # check mouse is below bottom of last drawn row so that clicking on the one pixel empty space between selectables
            # does not cause everything to unselect or popup to open
            if imgui.is_item_clicked(imgui.MouseButton_.left) and not any_selectable_clicked and imgui.get_io().mouse_pos.y>last_y:  # NB: table header is not signalled by is_item_clicked(), so this works correctly
                utils.set_all(self.selected_recordings, False)

            # show menu when right-clicking the empty space
            if self.empty_context_callback and imgui.get_io().mouse_pos.y>last_y and imgui.begin_popup_context_item("##recording_list_context",popup_flags=imgui.PopupFlags_.mouse_button_right | imgui.PopupFlags_.no_open_over_existing_popup):
                utils.set_all(self.selected_recordings, False)  # deselect on right mouse click as well
                self.empty_context_callback()
                imgui.end_popup()

    def remove_recording(self, iid: int):
        del self.recordings[iid]
        del self.selected_recordings[iid]

    def draw_eye_tracker_widget(self, rec: recording.Recording, align=False):
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
        iid         = imgui.get_id(rec.eye_tracker.value)
        label_size  = imgui.calc_text_size(rec.eye_tracker.value)
        size        = imgui.ImVec2(self._eye_tracker_label_width, label_size.y)
        pos         = imgui.get_cursor_screen_pos()
        bb          = imgui.internal.ImRect(pos, (pos.x+size.x, pos.y+size.y))
        imgui.internal.item_size(size, 0)
        # if visible
        if imgui.internal.item_add(bb, iid):
            # draw frame
            imgui.internal.render_frame(bb.min, bb.max, imgui.color_convert_float4_to_u32(rec.eye_tracker.color), True, imgui.get_style().frame_rounding)
            # draw text on top
            imgui.internal.render_text_clipped((bb.min.x+x_padding, bb.min.y), (bb.max.x-x_padding, bb.max.y), rec.eye_tracker.value, None, label_size, imgui.get_style().button_text_align, bb)

        if align:
            imgui.end_group()
        imgui.pop_style_var()

    def draw_recording_name_text(self, rec: recording.Recording, accent_color: tuple[float] = None):
        if accent_color is not None:
            imgui.text_colored(accent_color, rec.name)
        else:
            imgui.text(rec.name)

    def draw_working_directory(self, rec: recording.Recording):
        imgui.text(rec.working_directory.name if rec.working_directory else "Unknown")
        if imgui.is_item_hovered():
            if rec.working_directory and rec.working_directory.is_dir():
                text = str(rec.working_directory)
            else:
                text = 'Working directory not created yet'
            gui_utils.draw_tooltip(text)

    def draw_source_directory(self, rec: recording.Recording):
        imgui.text(rec.source_directory.stem or "Unknown")
        if rec.source_directory and imgui.is_item_hovered():
            gui_utils.draw_tooltip(str(rec.source_directory))

    def sort_and_filter_recordings(self, sort_specs_in: imgui.TableSortSpecs):
        if sort_specs_in.specs_dirty or self.require_sort:
            ids = list(self.recordings)
            sort_specs = [sort_specs_in.get_specs(i) for i in range(sort_specs_in.specs_count)]
            for sort_spec in reversed(sort_specs):
                key = self._columns[sort_spec.column_index].sort_key_func
                ids.sort(key=key, reverse=sort_spec.get_sort_direction()==imgui.SortDirection.descending)
            self.sorted_recordings_ids = ids
            for flt in self.filters:
                key = lambda iid: flt.invert != flt.fun(self.recordings[iid])
                if key is not None:
                    self.sorted_recordings_ids = list(filter(key, self.sorted_recordings_ids))
            if self.filter_box_text:
                search = self.filter_box_text.lower()
                def key(iid):
                    recording = self.recordings[iid]
                    return \
                        search in recording.eye_tracker.value.lower() or \
                        search in recording.name.lower() or \
                        search in recording.participant.lower() or \
                        search in recording.project.lower()
                self.sorted_recordings_ids = list(filter(key, self.sorted_recordings_ids))
            sort_specs_in.specs_dirty = False
            self.require_sort = False