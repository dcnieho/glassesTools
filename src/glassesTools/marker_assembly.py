import csv
import math
import pathlib

import cv2
import numpy as np

from . import marker


class MarkerAssembly:
    def __init__(self,
                 marker_corners: dict[int, np.ndarray],
                 aruco_dict_id: int = cv2.aruco.DICT_4X4_250,
                 aruco_settings: dict | None = None,
                 min_num_markers: int = 3,
                 unit: str | None = None):
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

    def set_origin(self, origin: np.ndarray):
        origin = np.asarray(origin, dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError('Marker assembly origin must be a finite 3D point')
        offset = origin - self._origin
        for corners in self.marker_corners.values():
            corners -= offset[:corners.shape[1]]
        self._origin = origin.copy()

    @classmethod
    def from_file(cls, marker_file: str | pathlib.Path, aruco_dict_id: int = cv2.aruco.DICT_4X4_250, aruco_settings: dict | None = None, min_num_markers: int = 3, unit: str | None = None):
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
        )

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