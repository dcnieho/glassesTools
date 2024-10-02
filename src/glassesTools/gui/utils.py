import random
import functools
import sys
import traceback
from typing import Any, Callable
from imgui_bundle import imgui, hello_imgui, icons_fontawesome_6 as ifa6
import glfw

from . import msg_box
from .. import platform, utils


# https://gist.github.com/Willy-JL/f733c960c6b0d2284bcbee0316f88878
def get_traceback(*exc_info: list):
    exc_info = exc_info or sys.exc_info()
    tb_lines = traceback.format_exception(*exc_info)
    tb = "".join(tb_lines)
    return tb


def draw_tooltip(hover_text):
    imgui.begin_tooltip()
    imgui.push_text_wrap_pos(min(imgui.get_font_size() * 35, imgui.get_io().display_size.x))
    imgui.text_unformatted(hover_text)
    imgui.pop_text_wrap_pos()
    imgui.end_tooltip()

def draw_hover_text(hover_text: str, text="(?)", force=False, hovered_flags=imgui.HoveredFlags_.for_tooltip|imgui.HoveredFlags_.delay_normal, *args, **kwargs):
    if text:
        imgui.text_disabled(text, *args, **kwargs)
    if force or imgui.is_item_hovered(hovered_flags):
        draw_tooltip(hover_text)
        return True
    return False


def rand_num_str(len=8):
    return "".join((random.choice('0123456789') for _ in range(len)))

def push_popup(gui, *args, bottom=False, **kwargs):
    if len(args) + len(kwargs) > 1:
        if args[0] in (popup, msg_box.msgbox):
            args = list(args)
            args[1] = args[1] + "##popup_" + rand_num_str()
        popup_func = functools.partial(*args, **kwargs)
    else:
        popup_func = args[0]
    if bottom:
        gui.popup_stack.insert(0, popup_func)
    else:
        gui.popup_stack.append(popup_func)
    return popup_func

def fix_popup_transparency():
    frame_bg_col = imgui.get_style().color_(imgui.Col_.title_bg_active)
    imgui.get_style().set_color_(imgui.Col_.title_bg_active,(frame_bg_col.x, frame_bg_col.y, frame_bg_col.z, 1.))
    popup_bg_col = imgui.get_style().color_(imgui.Col_.popup_bg)
    imgui.get_style().set_color_(imgui.Col_.popup_bg,(popup_bg_col.x, popup_bg_col.y, popup_bg_col.z, 1.))

def handle_popup_stack(popup_stack: list):
    fix_popup_transparency()
    open_popup_count = 0
    for popup in popup_stack:
        if hasattr(popup, "tick"):
            popup_func = popup.tick
        else:
            popup_func = popup
        opened, closed = popup_func()
        if closed:
            popup_stack.remove(popup)
        open_popup_count += opened
    # Popups are closed all at the end to allow stacking
    for _ in range(open_popup_count):
        imgui.end_popup()

def close_weak_popup(check_escape: bool = True, check_click_outside: bool = True):
    if not imgui.is_popup_open("", imgui.PopupFlags_.any_popup_id):
        # This is the topmost popup
        if check_escape and imgui.is_key_pressed(imgui.Key.escape):
            # Escape is pressed
            imgui.close_current_popup()
            return True
        elif check_click_outside and imgui.is_mouse_clicked(imgui.MouseButton_.left):
            # Mouse was just clicked
            pos = imgui.get_window_pos()
            size = imgui.get_window_size()
            if not imgui.is_mouse_hovering_rect(pos, (pos.x+size.x, pos.y+size.y), clip=False):
                # Popup is not hovered
                imgui.close_current_popup()
                return True
    return False

popup_flags: int = (
    imgui.WindowFlags_.no_collapse |
    imgui.WindowFlags_.no_saved_settings |
    imgui.WindowFlags_.always_auto_resize
)

def popup(label: str, popup_content: Callable, buttons: dict[str, Callable] = None, closable=True, escape=True, outside=True):
    if buttons is True:
        buttons = {
            ifa6.ICON_FA_CHECK + " Ok": None
        }
    if not imgui.is_popup_open(label):
        imgui.open_popup(label)
    closed = False
    opened = 1
    if imgui.begin_popup_modal(label, closable or None, flags=popup_flags)[0]:
        if escape or outside:
            closed = close_weak_popup(check_escape=escape, check_click_outside=outside)
        imgui.begin_group()
        activate_button = popup_content()
        imgui.end_group()
        imgui.spacing()
        if buttons:
            btns_width = sum(imgui.calc_text_size(name).x for name in buttons) + (2 * len(buttons) * imgui.get_style().frame_padding.x) + (imgui.get_style().item_spacing.x * (len(buttons) - 1))
            cur_pos_x = imgui.get_cursor_pos_x()
            new_pos_x = cur_pos_x + imgui.get_content_region_avail().x - btns_width
            if new_pos_x > cur_pos_x:
                imgui.set_cursor_pos_x(new_pos_x)
            for i, (label,callbacks) in enumerate(buttons.items()):
                if isinstance(callbacks,tuple):
                    callback = callbacks[0]
                    disabled = len(callbacks)>1 and callbacks[1]()
                else:
                    callback = callbacks
                    disabled = False
                if disabled:
                    imgui.begin_disabled()
                if imgui.button(label) or activate_button==i:
                    if callback:
                        callback()
                    imgui.close_current_popup()
                    closed = True
                if disabled:
                    imgui.end_disabled()
                imgui.same_line()
    else:
        opened = 0
        closed = True
    return opened, closed


def selectable_item_logic(iid: int|str, selected: dict[int,Any], last_clicked_id: int, sorted_ids: list[int],
                          selectable_clicked: bool, new_selectable_state: bool,
                          allow_multiple=True, overlayed_hovered=False, overlayed_clicked=False, new_overlayed_state=False):
    if overlayed_clicked:
        if not allow_multiple:
            utils.set_all(selected, False)
        selected[iid] = new_overlayed_state
        last_clicked_id = iid
    elif selectable_clicked and not overlayed_hovered: # don't enter this branch if interaction is with another overlaid actionable item
        if not allow_multiple:
            utils.set_all(selected, False)
            selected[iid] = new_selectable_state
        else:
            num_selected = sum([selected[id] for id in sorted_ids])
            if not imgui.get_io().key_ctrl:
                # deselect all, below we'll either select all, or range between last and current clicked
                utils.set_all(selected, False)

            if imgui.get_io().key_shift:
                # select range between last clicked and just clicked item
                idx              = sorted_ids.index(iid)
                last_clicked_idx = sorted_ids.index(last_clicked_id)
                idxs = sorted([idx, last_clicked_idx])
                for rid in range(idxs[0],idxs[1]+1):
                    selected[sorted_ids[rid]] = True
            else:
                selected[iid] = True if num_selected>1 and not imgui.get_io().key_ctrl else new_selectable_state

            # consistent with Windows behavior, only update last clicked when shift not pressed
            if not imgui.get_io().key_shift:
                last_clicked_id = iid

    return last_clicked_id

def handle_item_hitbox_events(iid: int, selected: dict[int,bool], context_menu: Callable[[int],bool]) -> tuple[bool,bool]:
    right_clicked = False
    selectables_edited = False
    # Right click = context menu
    if context_menu and imgui.begin_popup_context_item(f"##{iid}_context"):
        right_clicked = True
        selectables_edited = context_menu(iid)
        imgui.end_popup()
    else:
        right_clicked = imgui.is_item_clicked(imgui.MouseButton_.right)

    if right_clicked:
        # update selected items. same logic as windows explorer:
        # 1. if right-clicked on one of the selected items, regardless of what modifier is pressed, keep selection as is
        # 2. if right-clicked elsewhere than on one of the selected items:
        # 2a. if control is down pop up right-click menu for the selected items.
        # 2b. if control not down, deselect everything except clicked item (if any)
        # NB: popup not shown when shift or control are down, do not know why...
        if not selected[iid] and not imgui.get_io().key_ctrl:
            utils.set_all(selected, False)
            selected[iid] = True
    return right_clicked, selectables_edited


def my_checkbox(label: str, state: bool, frame_size: tuple=None, frame_padding_override: list=None, do_vertical_align=True):
    style = imgui.get_style()
    if state:
        imgui.push_style_color(imgui.Col_.frame_bg_hovered, style.color_(imgui.Col_.button_hovered))
        imgui.push_style_color(imgui.Col_.frame_bg, style.color_(imgui.Col_.button_hovered))
        imgui.push_style_color(imgui.Col_.check_mark, style.color_(imgui.Col_.text))
    if frame_size is not None:
        frame_padding = frame_padding_override if frame_padding_override else [style.frame_padding.x, style.frame_padding.y]
        imgui.push_style_var(imgui.StyleVar_.frame_padding, frame_size)
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0.,0.))
        imgui.begin_group()
        if do_vertical_align and frame_padding[1]:
            imgui.dummy((0,frame_padding[1]))
        imgui.dummy((frame_padding[0],0))
        imgui.same_line()
    result = imgui.checkbox(label, state)
    if frame_size is not None:
        imgui.end_group()
        imgui.pop_style_var(2)
    if state:
        imgui.pop_style_color(3)
    return result

def my_combo(*args, **kwargs):
    imgui.push_style_color(imgui.Col_.button, imgui.get_style().color_(imgui.Col_.button_hovered))
    result = imgui.combo(*args, **kwargs)
    imgui.pop_style_color()
    return result


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