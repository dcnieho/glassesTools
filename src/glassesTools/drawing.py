import numpy as np
import math
import cv2
import typing

from . import json, marker, ocv, transforms


class RgbColor(typing.NamedTuple):
    r: int = 0
    g: int = 0
    b: int = 0


class CircleStyle(typing.NamedTuple):
    radius: int = 8
    edge_color: RgbColor | None = RgbColor(0, 255, 0)
    edge_thickness: int = 2
    fill_color: RgbColor | None = None


DEFAULT_CIRCLE_STYLE = CircleStyle()


def draw_circle(img: np.ndarray, center, style: CircleStyle | None = None, sub_pixel_fac=1):
    style = style or DEFAULT_CIRCLE_STYLE
    if style.fill_color is not None:
        openCVCircle(img, center, style.radius, style.fill_color[::-1], -1, sub_pixel_fac)
    if style.edge_color is not None and style.edge_thickness > 0:
        openCVCircle(img, center, style.radius, style.edge_color[::-1], style.edge_thickness, sub_pixel_fac)


class MarkerIDStyle(typing.NamedTuple):
    draw: bool = True
    color: RgbColor | None = None
    background_opacity: float = 0.0
    background_color: RgbColor | None = None
    font_size: float = 0.8


class MarkerStyle(typing.NamedTuple):
    border_color: RgbColor | None = RgbColor(0, 255, 0)
    corner_color: RgbColor | None = None
    id_style: MarkerIDStyle = MarkerIDStyle()


class MarkerStyles(typing.NamedTuple):
    assembly: MarkerStyle = MarkerStyle()
    individual: MarkerStyle = MarkerStyle(border_color=RgbColor(255, 0, 255))
    unexpected: MarkerStyle = MarkerStyle(border_color=RgbColor(255, 255, 128))


for style_type, compatible_names in (
    (CircleStyle, None),
    (RgbColor, ['__config.RgbColor__']),
    (MarkerIDStyle, None),
    (MarkerStyle, None),
    (MarkerStyles, None),
):
    json.register_type(json.TypeEntry(
        style_type,
        f'glassesTools.drawing.{style_type.__name__}',
        lambda value: value._asdict(),
        lambda value, style_type=style_type: style_type(**value),
        compatible_names,
    ))

DEFAULT_MAPPED_MARKER_STYLES = MarkerStyles(
    assembly=MarkerStyle(border_color=RgbColor(0, 255, 0), id_style=MarkerIDStyle(background_opacity=0.4)),
    individual=MarkerStyle(border_color=RgbColor(255, 0, 255), id_style=MarkerIDStyle(background_opacity=0.4)),
    unexpected=MarkerStyle(border_color=RgbColor(255, 255, 128), id_style=MarkerIDStyle(background_opacity=0.4)),
)

def openCVCircle(img, center_coordinates, radius, color, thickness, sub_pixel_fac):
    p = [np.round(x*sub_pixel_fac) for x in center_coordinates]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(np.intc).max for x in p]):
        p = tuple([int(x) for x in p])
        cv2.circle(img, p, int(np.round(radius*sub_pixel_fac)), color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(sub_pixel_fac)))

def openCVLine(img, start_point, end_point, color, thickness, sub_pixel_fac):
    sp = [np.round(x*sub_pixel_fac) for x in start_point]
    ep = [np.round(x*sub_pixel_fac) for x in   end_point]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(np.intc).max for x in sp]) and np.all([not math.isnan(x) and abs(x)<np.iinfo(np.intc).max for x in ep]):
        sp = tuple([int(x) for x in sp])
        ep = tuple([int(x) for x in ep])
        cv2.line(img, sp, ep, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(sub_pixel_fac)))

def openCVRectangle(img, p1, p2, color, thickness, sub_pixel_fac):
    p1 = [np.round(x*sub_pixel_fac) for x in p1]
    p2 = [np.round(x*sub_pixel_fac) for x in p2]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(np.intc).max for x in p1]) and np.all([not math.isnan(x) and abs(x)<np.iinfo(np.intc).max for x in p2]):
        p1 = tuple([int(x) for x in p1])
        p2 = tuple([int(x) for x in p2])
        cv2.rectangle(img, p1, p2, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(sub_pixel_fac)))

def openCVPolylines(img, pts, isClosed, color, thickness, sub_pixel_fac):
    pts = np.round(pts*sub_pixel_fac).astype(np.int32)
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(np.intc).max for x in pts.flatten()]):
        cv2.polylines(img, [pts], isClosed, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(sub_pixel_fac)))

def openCVFrameAxis(img, cam_params: ocv.CameraParams, rvec,  tvec,  arm_length, thickness, sub_pixel_fac, position = [0.,0.,0.], ROI_offset=(0,0)):
    # same as the openCV function, but with anti-aliasing for a nicer image if subPixelFac>1
    points = np.vstack((np.zeros((1,3)), arm_length*np.eye(3)))+np.vstack(4*[np.asarray(position)])
    cam_points = transforms.project_points(points, cam_params, rot_vec=rvec, trans_vec=tvec, ROI_offset=ROI_offset)
    # z-sort them
    RMat = cv2.Rodrigues(rvec)[0]
    RtMat = np.hstack((RMat, tvec.reshape(3,1)))
    world_points = np.matmul(RtMat,cv2.convertPointsToHomogeneous(points[1:,:]).reshape((-1,4)).T)
    order = np.argsort(world_points[-1,:])[::-1]
    # draw
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    for i in order:
        openCVLine(img, cam_points[0].flatten(), cam_points[i+1].flatten(), colors[i], thickness, sub_pixel_fac)

def arucoDetectedMarkers(img, corners, ids, style: MarkerStyle|None=None, sub_pixel_fac=1, special_highlight=None):
    if special_highlight is None:
        special_highlight = []
    style = style or MarkerStyle()
    id_style = style.id_style
    if not 0.0 <= id_style.background_opacity <= 1.0:
        raise ValueError('Marker ID background opacity must be between 0 and 1')

    border_color = None if style.border_color is None else style.border_color[::-1]
    corner_color = style.corner_color
    if corner_color is None and style.border_color is not None:
        # corner color just swap G and R
        corner_color = RgbColor(style.border_color.g, style.border_color.b, style.border_color.r)
    corner_color = None if corner_color is None else corner_color[::-1]
    draw_IDs = id_style.draw and (ids is not None) and len(ids) > 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = id_style.font_size
    thickness = 2
    text_base_color = border_color or (0, 255, 0)
    text_color = id_style.color[::-1] if id_style.color is not None else tuple(255 - component for component in text_base_color)
    backing_color = id_style.background_color[::-1] if id_style.background_color is not None else tuple(255 - component for component in text_color)
    for i in range(0, len(corners)):
        corner = corners[i][0]
        # draw marker sides
        side_color = border_color
        for marker_ids, highlight_color in zip(special_highlight[::2], special_highlight[1::2]):
            if marker_ids is not None and ids is not None and ids[i] in marker_ids:
                side_color = highlight_color[::-1]
        if side_color is not None:
            for j in range(4):
                p0 = corner[j, :]
                p1 = corner[(j + 1) % 4, :]
                openCVLine(img, p0, p1, side_color, 1, sub_pixel_fac)

        # draw first corner mark
        if corner_color is not None:
            openCVRectangle(img, corner[0] - 3, corner[0] + 3, corner_color, 1, sub_pixel_fac)

        # draw IDs if wanted
        if draw_IDs and ids is not None:
            # get text size
            text = str(ids[i])
            (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
            # get marker center
            center = marker.corners_intersection(corner).astype(int)
            # determine text origin so its centered on the marker
            origin = (int(center[0] - width / 2), int(center[1] + (height - baseline) / 2))

            if id_style.background_opacity > 0:
                padding = 2
                box_size = max(width, height + baseline) + 2 * padding
                box_left = round(origin[0] + width / 2 - box_size / 2)
                box_top = round(origin[1] + (baseline - height) / 2 - box_size / 2)
                x0 = max(0, box_left)
                y0 = max(0, box_top)
                x1 = min(img.shape[1], box_left + box_size)
                y1 = min(img.shape[0], box_top + box_size)
                if x0 < x1 and y0 < y1:
                    roi = img[y0:y1, x0:x1]
                    background = np.empty_like(roi)
                    background[:] = backing_color
                    cv2.addWeighted(roi, 1-id_style.background_opacity, background, id_style.background_opacity, 0, dst=roi)
            cv2.putText(img, text, origin, font, scale, text_color, thickness, lineType=cv2.LINE_AA)