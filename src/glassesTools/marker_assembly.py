import csv
import math
import pathlib

import cv2
import numpy as np

from . import marker


class MarkerAssembly:
    default_ref_image_name = 'reference_image.png'

    def __init__(self,
                 marker_corners         : dict[int, np.ndarray],
                 aruco_dict_id          : int                       = cv2.aruco.DICT_4X4_250,
                 aruco_settings         : dict | None               = None,
                 min_num_markers        : int                       = 3,
                 unit                   : str | None                = None,
                 ref_image_store_path   : str | pathlib.Path | None = None,
                 ref_image_size         : int | None                = None
                 ) -> None:
        if not marker_corners:
            raise ValueError('A marker assembly must contain at least one marker')
        if len(marker_corners) != len(set(marker_corners)):
            raise ValueError('Marker IDs must be unique within an assembly')

        points = np.asarray(list(marker_corners.values()), dtype=np.float32)
        if points.ndim != 3 or points.shape[1] != 4 or points.shape[2] not in (2, 3):
            raise ValueError('Marker corners must have shape (N, 4, 2) or (N, 4, 3)')
        if not np.isfinite(points).all():
            raise ValueError('Marker corners must contain only finite coordinates')

        self.marker_corners = {int(marker_id): corners for marker_id, corners in zip(marker_corners, points)}
        self._all_marker_ids = list(self.marker_corners)
        self._origin = np.zeros(3, dtype=float)
        self.aruco_dict_id = aruco_dict_id
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        from . import aruco
        self.aruco_settings = aruco.resolve_settings(aruco_settings)
        self.min_num_markers = min_num_markers
        self.unit = unit
        self._ref_image_size = ref_image_size or 1920
        self._ref_image_cache: dict[int, np.ndarray] = {}
        if ref_image_size is not None:
            image = self._store_reference_image(ref_image_store_path, ref_image_size)
            self._ref_image_cache[ref_image_size] = image

    def set_origin(self, origin: np.ndarray):
        origin = np.asarray(origin, dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError('Marker assembly origin must be a finite 3D point')
        offset = origin - self._origin
        for corners in self.marker_corners.values():
            corners -= offset[:corners.shape[1]]
        self._origin = origin.copy()

    @classmethod
    def from_file(cls, marker_file: str | pathlib.Path, aruco_dict_id: int = cv2.aruco.DICT_4X4_250, aruco_settings: dict | None = None, min_num_markers: int = 3, unit: str | None = None, ref_image_store_path: str | pathlib.Path | None = None, ref_image_size: int | None = None):
        marker_file = pathlib.Path(marker_file)
        marker_corners: dict[int, list[list[float] | None]] = {}
        with marker_file.open('r', newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ['ID', 'corner', 'x', 'y', 'z']:
                raise ValueError(f'{marker_file}: expected columns ID,corner,x,y,z in that order')
            for line_number, row in enumerate(reader, start=2):
                try:
                    marker_id = int(row['ID'])
                    corner_idx = int(row['corner'])
                    point = [float(row[axis]) for axis in ('x', 'y', 'z')]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'{marker_file}:{line_number}: invalid marker ID, corner index, or coordinate') from exc
                if corner_idx not in range(4) or not all(math.isfinite(value) for value in point):
                    raise ValueError(f'{marker_file}:{line_number}: corner must be 0-3 and coordinates finite')
                corners = marker_corners.setdefault(marker_id, [None] * 4)
                if corners[corner_idx] is not None:
                    raise ValueError(f'{marker_file}:{line_number}: duplicate corner {corner_idx} for marker {marker_id}')
                corners[corner_idx] = point

        if not marker_corners or any(any(point is None for point in corners) for corners in marker_corners.values()):
            raise ValueError(f'{marker_file}: each marker ID must have exactly one row for each corner 0-3')
        return cls(
            {marker_id: corners for marker_id, corners in marker_corners.items()},
            aruco_dict_id=aruco_dict_id,
            aruco_settings=aruco_settings,
            min_num_markers=min_num_markers,
            unit=unit,
            ref_image_store_path=ref_image_store_path,
            ref_image_size=ref_image_size,
        )

    def get_ref_image(self, im_size: int | None = None, as_RGB: bool = False) -> np.ndarray:
        if im_size is None:
            im_size = self._ref_image_size
        if not isinstance(im_size, int):
            raise TypeError(f'image size should be an int, not {type(im_size)}')
        if not self._ref_image_cache:
            self._ref_image_cache[self._ref_image_size] = self._store_reference_image(None, self._ref_image_size)
        if im_size not in self._ref_image_cache:
            scale = float(im_size) / self._ref_image_size
            self._ref_image_cache[im_size] = cv2.resize(self._ref_image_cache[self._ref_image_size], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        image = self._ref_image_cache[im_size]
        if as_RGB:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.copy()

    def _store_reference_image(self, path: str | pathlib.Path | None, im_size: int) -> np.ndarray:
        image = self._generate_reference_image(im_size)
        if path:
            cv2.imwrite(str(path), image)
        return image

    def _generate_reference_image(self, im_size: int) -> np.ndarray:
        # set up a white canvas
        tile_size = im_size // 2
        canvas = np.full((tile_size * 2, tile_size * 2, 3), 255, dtype=np.uint8)
        # get marker images
        marker_render_size = 256
        marker_images = {
            marker_id: self.aruco_dict.generateImageMarker(
                marker_id, marker_render_size,
                borderBits=self.aruco_settings['detector_params']['markerBorderBits'])
            for marker_id in self.marker_corners
        }
        coverage_source = np.full((marker_render_size, marker_render_size), 255, dtype=np.uint8)
        marker_edge = marker_render_size - 1
        source = np.array([[0, 0], [marker_edge, 0], [marker_edge, marker_edge], [0, marker_edge]], dtype=np.float32)
        # get marker positions
        corners_by_id = {}
        all_points = []
        for marker_id, corners in self.marker_corners.items():
            corners_3d = np.asarray(corners, dtype=np.float32)
            if corners_3d.shape[1] == 2:
                corners_3d = np.column_stack((corners_3d, np.zeros(4, dtype=np.float32)))
            corners_by_id[marker_id] = corners_3d
            all_points.append(corners_3d)
        all_points = np.concatenate(all_points)
        target = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
        relative_points = all_points - target
        # get marker normals and corner positions
        marker_geometry = {}
        for marker_id, corners_3d in corners_by_id.items():
            normal = np.cross(corners_3d[1] - corners_3d[0], corners_3d[3] - corners_3d[0])
            normal_length = np.linalg.norm(normal)
            normal = None if normal_length == 0 else normal / normal_length
            marker_geometry[marker_id] = (corners_3d - target, normal)
        max_extent = max(float(np.ptp(all_points, axis=0).max()), 1e-6)
        # determine camera views (positions and orientations) for each of the four panels
        isometric_elevation = np.deg2rad(60)
        isometric_azimuth = np.deg2rad(45)
        isometric_forward = np.array([
            np.cos(isometric_elevation) * np.cos(isometric_azimuth),
            -np.cos(isometric_elevation) * np.sin(isometric_azimuth),
            np.sin(isometric_elevation),
        ])
        views = (
            (  isometric_forward   , np.array([0., -1., 0.]), None),
            (np.array([0., 0., 1.]), np.array([0., -1., 0.]), 2),
            (np.array([0., 1., 0.]), np.array([0.,  0., 1.]), 1),
            (np.array([1., 0., 0.]), np.array([0., -1., 0.]), 0),
        )
        # set up axes and origin
        axis_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
        axis_labels = ('+X', '+Y', '+Z')
        model_origin = np.zeros(3, dtype=np.float32)
        axis_world_length = max_extent * 0.1
        axis_fit_points = np.vstack((model_origin, np.eye(3) * axis_world_length))
        fit_relative_points = np.concatenate((relative_points, axis_fit_points - target))
        # set up projection
        tan_half_fov = np.tan(np.deg2rad(55 / 2))
        half_canvas = max(tile_size / 2 - 8, 1)
        focal_length = half_canvas / tan_half_fov
        # render each view
        for view_idx, (camera_forward, camera_up, forward_axis) in enumerate(views):
            # compute model space points and project into camera space
            camera_forward = camera_forward / np.linalg.norm(camera_forward)
            right = np.cross(camera_forward, camera_up)
            right /= np.linalg.norm(right)
            down = np.cross(camera_forward, right)
            camera_points = np.column_stack((fit_relative_points @ right,
                                             fit_relative_points @ down,
                                             fit_relative_points @ camera_forward))
            # determine camera distance to all points fit in view
            fit_distance = np.max(np.maximum(np.abs(camera_points[:, 0]), np.abs(camera_points[:, 1])) / tan_half_fov - camera_points[:, 2])
            min_distance = -float(camera_points[:, 2].min()) + max_extent * 0.1
            camera_distance = max(float(fit_distance), min_distance, max_extent * 0.5)
            visible = []
            for marker_id, (relative_corners, normal) in marker_geometry.items():
                # determine if marker is facing camera
                if normal is None or np.dot(normal, camera_forward) <= 1e-6:
                    # not facing camera, so skip it
                    continue
                # project marker corners into camera space and then into image space
                camera_x = relative_corners @ right
                camera_y = relative_corners @ down
                camera_z = camera_distance + relative_corners @ camera_forward
                projected = np.column_stack((tile_size / 2 + focal_length * camera_x / camera_z,
                                             tile_size / 2 + focal_length * camera_y / camera_z))
                visible.append((float(np.mean(camera_z)), marker_id, projected))
            if visible:
                # actually draw the markers in this view, sorted by depth so that closer markers are drawn on top of farther ones
                y_slice = slice((view_idx // 2) * tile_size, (view_idx // 2 + 1) * tile_size)
                x_slice = slice((view_idx % 2) * tile_size, (view_idx % 2 + 1) * tile_size)
                tile = canvas[y_slice, x_slice]
                for _, marker_id, projected in sorted(visible, key=lambda entry: entry[0], reverse=True):
                    x_min = max(0, int(np.floor(projected[:, 0].min())) - 2)
                    y_min = max(0, int(np.floor(projected[:, 1].min())) - 2)
                    x_max = min(tile_size, int(np.ceil(projected[:, 0].max())) + 3)
                    y_max = min(tile_size, int(np.ceil(projected[:, 1].max())) + 3)
                    if x_min >= x_max or y_min >= y_max:
                        continue

                    destination = projected.astype(np.float32)
                    destination[:, 0] -= x_min
                    destination[:, 1] -= y_min
                    transform = cv2.getPerspectiveTransform(source, destination)
                    roi_size = (x_max - x_min, y_max - y_min)
                    warped = cv2.warpPerspective(marker_images[marker_id], transform, roi_size, flags=cv2.INTER_AREA, borderValue=255)
                    coverage = cv2.warpPerspective(coverage_source, transform, roi_size, flags=cv2.INTER_AREA, borderValue=0)
                    alpha = coverage.astype(np.float32)[..., None] / 255
                    region = tile[y_min:y_max, x_min:x_max]
                    region[:] = np.clip(region.astype(np.float32) * (1 - alpha) + warped.astype(np.float32)[..., None] * alpha, 0, 255).astype(np.uint8)
            # draw axes and labels. Axes originate at the model origin
            if tile_size >= 32:
                panel_x = (view_idx % 2) * tile_size
                panel_y = (view_idx // 2) * tile_size
                origin_relative = model_origin - target
                origin_depth = camera_distance + np.dot(origin_relative, camera_forward)
                origin = np.rint((
                    panel_x + tile_size / 2 + focal_length * np.dot(origin_relative, right) / origin_depth,
                    panel_y + tile_size / 2 + focal_length * np.dot(origin_relative, down) / origin_depth,
                )).astype(int)
                font_scale = min(1.5, max(0.5, tile_size / 384))
                text_thickness = max(1, round(font_scale))
                for axis_idx, (axis, axis_label, color) in enumerate(zip(np.eye(3), axis_labels, axis_colors)):
                    if forward_axis is not None and axis_idx == forward_axis:
                        continue
                    axis_tip_relative = axis * axis_world_length - target
                    tip_depth = camera_distance + np.dot(axis_tip_relative, camera_forward)
                    endpoint = np.rint((
                        panel_x + tile_size / 2 + focal_length * np.dot(axis_tip_relative, right) / tip_depth,
                        panel_y + tile_size / 2 + focal_length * np.dot(axis_tip_relative, down) / tip_depth,
                    )).astype(int)
                    direction = endpoint - origin
                    projected_length = np.linalg.norm(direction)
                    if projected_length < 1:
                        continue
                    direction = direction / projected_length
                    head_angle = np.pi / 4
                    head_length = projected_length * 0.25
                    cv2.line(canvas, tuple(origin), tuple(endpoint), color, 2, cv2.LINE_AA)
                    for wing_angle in (-head_angle, head_angle):
                        wing_direction = np.array([
                            -direction[0] * np.cos(wing_angle) + direction[1] * np.sin(wing_angle),
                            -direction[0] * np.sin(wing_angle) - direction[1] * np.cos(wing_angle),
                        ])
                        wing_end = (int(endpoint[0] + wing_direction[0] * head_length),
                                    int(endpoint[1] + wing_direction[1] * head_length))
                        cv2.line(canvas, tuple(endpoint), wing_end, color, 2, cv2.LINE_AA)
                    text_size, _ = cv2.getTextSize(axis_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                    label_center = (endpoint[0] + direction[0] * (text_size[0] / 2 + 4),
                                    endpoint[1] + direction[1] * (text_size[1] / 2 + 4))
                    label_origin = (int(label_center[0] - text_size[0] / 2),
                                    int(label_center[1] + text_size[1] / 2))
                    cv2.putText(canvas, axis_label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)
        cv2.line(canvas, (tile_size, 0), (tile_size, tile_size * 2 - 1), (192, 192, 192), 1)
        cv2.line(canvas, (0, tile_size), (tile_size * 2 - 1, tile_size), (192, 192, 192), 1)
        return canvas

    def get_marker_IDs(self) -> dict[str|int,list[marker.MarkerID]]:
        return {'assembly': [marker.MarkerID(marker_id, self.aruco_dict_id) for marker_id in self._all_marker_ids]}

    def is_dynamic(self) -> bool:
        return False

    def get_dynamic_marker_setup(self) -> 'aruco.MarkerSetup | None':
        return None

    def get_aruco_board(self) -> cv2.aruco.Board:
        from . import aruco
        return aruco.create_assembly_geometry(list(self.marker_corners.values()), self._all_marker_ids, self.aruco_dict)

    def get_assembly_setup(self):
        from . import aruco
        marker_ids = self.get_marker_IDs()
        is_dynamic = self.is_dynamic()
        return aruco.AssemblySetup(
            aruco_board     = self.get_aruco_board(),
            aruco_dict_id   = self.aruco_dict_id,
            marker_ids      = marker_ids,
            min_num_markers = self.min_num_markers,
            aruco_settings  = self.aruco_settings,
            unit            = self.unit,
            is_dynamic      = is_dynamic,
            dynamic_marker_ids = [mark for group, marks in marker_ids.items() if group != 'assembly' for mark in marks],
            dynamic_marker_setup = self.get_dynamic_marker_setup() if is_dynamic else None,
        )