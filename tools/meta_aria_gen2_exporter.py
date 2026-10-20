"""Export Aria Gen 2 RGB video, camera calibration, and MPS eye gaze.

pip install projectaria-tools>=2.3.0 pycolmap ffmpeg-binaries pandas opencv-python-headless

By default, gaze is read from the on-device VRS eye-gaze stream (373-1).
Set USE_MPS_GAZE = True to use MPS gaze instead. In that mode, provide an MPS
output folder containing general_eye_gaze.csv or personalized_eye_gaze.csv;
personalized gaze is preferred when both are available.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import ffmpeg
import numpy as np
import pandas as pd
import pycolmap

from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps import MpsDataPathsProvider


# ---------------------------- User configuration ----------------------------
vrs_file = Path(r"~/Downloads/aria/Test_20261020_152335.vrs")
mps_folder: Path | None = None  # Defaults to mps_<vrs-stem>_vrs beside the VRS.
output_folder: Path | None = None  # Defaults to export_<vrs-stem> beside the VRS.
USE_MPS_GAZE = False  # If True, require and export MPS gaze instead of on-device gaze.

# Gen 2 VRS stream IDs: RGB camera 214-1; left eye-tracking camera 211-1.
RGB_STREAM_ID = StreamId("214-1")
ET_LEFT_STREAM_ID = StreamId("211-1")


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def as_xyz(value) -> np.ndarray:
    """Convert a Project Aria vector binding to a float64 NumPy 3-vector."""
    return np.asarray(value, dtype=np.float64).reshape(3)


def unit_gaze_direction(yaw: float, pitch: float) -> np.ndarray:
    """Return one CPF gaze direction, using Project Aria's yaw/pitch convention."""
    return as_xyz(mps.get_eyegaze_point_at_depth(float(yaw), float(pitch), 1.0))


def main() -> None:
    ffmpeg.add_to_path()
    vrs_path = vrs_file.expanduser().resolve()
    require_file(vrs_path, "VRS recording")

    if output_folder is None:
        out_dir = vrs_path.parent / f"export_{vrs_path.stem}"
    else:
        out_dir = output_folder.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating data provider from {vrs_path}")
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"Could not open VRS file: {vrs_path}")

    print("The following streams are present in this recording:")
    streams = provider.get_all_streams()
    for stream_id in streams:
        print(f"  stream_id: {stream_id} [{provider.get_label_from_stream_id(stream_id)}]")

    for stream_id, label in ((RGB_STREAM_ID, "RGB camera"), (ET_LEFT_STREAM_ID, "left eye camera")):
        if stream_id not in streams:
            raise RuntimeError(f"No {label} stream ({stream_id}) found in this recording.")

    rgb_label = provider.get_label_from_stream_id(RGB_STREAM_ID)
    et_label = provider.get_label_from_stream_id(ET_LEFT_STREAM_ID)
    rgb_start_ns = provider.get_first_time_ns(RGB_STREAM_ID, TimeDomain.DEVICE_TIME)
    rgb_end_ns = provider.get_last_time_ns(RGB_STREAM_ID, TimeDomain.DEVICE_TIME)
    et_start_ns = provider.get_first_time_ns(ET_LEFT_STREAM_ID, TimeDomain.DEVICE_TIME)
    et_end_ns = provider.get_last_time_ns(ET_LEFT_STREAM_ID, TimeDomain.DEVICE_TIME)
    print(f"RGB stream: {(rgb_end_ns - rgb_start_ns) / 1e9:.3f} s ({rgb_start_ns}--{rgb_end_ns} ns)")
    print(f"ET-left stream: {(et_end_ns - et_start_ns) / 1e9:.3f} s ({et_start_ns}--{et_end_ns} ns)")
    print(
        "RGB minus ET-left onset/end: "
        f"{(rgb_start_ns - et_start_ns) / 1e6:.3f}, "
        f"{(rgb_end_ns - et_end_ns) / 1e6:.3f} ms"
    )

    def print_image_config(config) -> None:
        for field in (
            "device_type", "device_version", "device_serial", "sensor_serial",
            "nominal_rate_hz", "image_width", "image_height", "pixel_format", "gamma_factor",
        ):
            if hasattr(config, field):
                print(f"  {field} {getattr(config, field)}")

    rgb_config = provider.get_image_configuration(RGB_STREAM_ID)
    et_config = provider.get_image_configuration(ET_LEFT_STREAM_ID)
    print("RGB camera image configuration:")
    print_image_config(rgb_config)
    print("Left eye camera image configuration:")
    print_image_config(et_config)

    # Export the RGB camera stream with Project Aria Tools' supported CLI.
    # The old Python helper import (vrs_to_mp4_utils) is not present in some
    # current Gen 2 package builds, while the vrs_to_mp4 executable is.
    video_path = out_dir / "worldCamera.mp4"
    converter = shutil.which("vrs_to_mp4")
    if converter is None:
        raise RuntimeError(
            "Could not find the 'vrs_to_mp4' command. Activate the environment "
            "where projectaria-tools is installed and check that its scripts "
            "directory is on PATH."
        )
    subprocess.run(
        [
            converter,
            "--vrs", str(vrs_path),
            "--output_video", str(video_path),
            "--stream_id", "214-1",
        ],
        check=True,
    )

    # Gen 2 records its ML eye-gaze result directly in the VRS. Use that as
    # the default source so the exported gaze matches a VRS-only Rerun view.
    # MPS is an explicitly selected alternative and must be present when asked for.
    gaze_source_path: Path | None = None
    if USE_MPS_GAZE:
        if mps_folder is None:
            mps_dir = vrs_path.parent / f"mps_{vrs_path.stem}_vrs"
        else:
            mps_dir = mps_folder.expanduser().resolve()
        if not mps_dir.is_dir():
            raise FileNotFoundError(
                f"USE_MPS_GAZE=True, but the MPS output folder does not exist: {mps_dir}"
            )

        mps_paths = MpsDataPathsProvider(str(mps_dir)).get_data_paths()
        gaze_paths = mps_paths.eyegaze
        personalized_value = getattr(gaze_paths, "personalized_eyegaze", None)
        general_value = getattr(gaze_paths, "general_eyegaze", None)
        personalized_path = Path(str(personalized_value)) if personalized_value else None
        general_path = Path(str(general_value)) if general_value else None

        if personalized_path is not None and personalized_path.is_file():
            gaze_source_path = personalized_path
            gaze_source = "MPS personalized gaze"
        elif general_path is not None and general_path.is_file():
            gaze_source_path = general_path
            gaze_source = "MPS general gaze"
        else:
            raise FileNotFoundError(
                "USE_MPS_GAZE=True, but neither eye_gaze/personalized_eye_gaze.csv "
                f"nor eye_gaze/general_eye_gaze.csv exists under {mps_dir}"
            )

        selected_gaze = mps.read_eyegaze(str(gaze_source_path))
        if not selected_gaze:
            raise RuntimeError(f"MPS gaze file contains no samples: {gaze_source_path}")
        print(f"Using {gaze_source}: {gaze_source_path}")
    else:
        gaze_stream_id = StreamId("373-1")
        if gaze_stream_id not in streams:
            raise RuntimeError(
                "On-device gaze was selected, but VRS eye-gaze stream 373-1 "
                "is not present. Set USE_MPS_GAZE=True to request MPS gaze instead."
            )
        gaze_count = provider.get_num_data(gaze_stream_id)
        selected_gaze = [
            provider.get_eye_gaze_data_by_index(gaze_stream_id, index)
            for index in range(gaze_count)
        ]
        selected_gaze = [gaze for gaze in selected_gaze if gaze is not None]
        if not selected_gaze:
            raise RuntimeError("On-device gaze stream 373-1 contains no readable samples.")
        gaze_source = "on-device VRS gaze (373-1)"
        print(f"Using {gaze_source}: {len(selected_gaze)} samples")

    device_calibration = provider.get_device_calibration()
    rgb_calib = device_calibration.get_camera_calib(rgb_label)

    # Gen 2 RGB uses the FisheyeRadTanThinPrism (Fisheye624) projection model.
    # Its COLMAP equivalent is RAD_TAN_THIN_PRISM_FISHEYE.
    model_name = rgb_calib.get_model_name()
    if model_name != calibration.CameraModelType.FISHEYE624:
        raise ValueError(
            "Expected the Gen 2 RGB Fisheye624/FisheyeRadTanThinPrism model, "
            f"but found {model_name}."
        )

    # vrs_to_mp4 exports the Gen 2 RGB stream in the recording's native
    # orientation; keep the calibration in that same orientation.
    rgb_calib_for_video = rgb_calib

    # Build the COLMAP camera parameters from the Aria calibration.
    image_width, image_height = rgb_calib_for_video.get_image_size()
    colmap_camera = pycolmap.Camera.create_from_model_id(
        0,
        pycolmap.CameraModelId.RAD_TAN_THIN_PRISM_FISHEYE,
        float(rgb_calib_for_video.get_focal_lengths()[0]),
        int(image_width),
        int(image_height),
    )
    params = np.zeros((colmap_camera.extra_params_idxs()[-1] + 1,), dtype=np.float64)
    params[colmap_camera.focal_length_idxs()] = rgb_calib_for_video.get_focal_lengths()
    params[colmap_camera.principal_point_idxs()] = rgb_calib_for_video.get_principal_point()
    extra_indices = colmap_camera.extra_params_idxs()
    projection_params = rgb_calib_for_video.get_projection_params()
    params[extra_indices] = projection_params[-len(extra_indices):]
    colmap_camera.params = params

    # Use the default Gen 2 CPF and sensor transforms, consistent with the
    # Gen 2 gaze reprojection helper (rather than pinning the Gen 1 CAD CPF).
    T_device_cpf = device_calibration.get_transform_device_cpf()
    T_device_rgb = device_calibration.get_transform_device_sensor(rgb_label)
    T_rgb_cpf = T_device_rgb.inverse() @ T_device_cpf

    camera_info = {
        "resolution": rgb_calib_for_video.get_image_size(),
        "position": T_rgb_cpf.to_quat_and_translation()[0][4:] * 1000.0,
        "rotation": T_rgb_cpf.to_matrix()[:3, :3],
        "colmap_camera": colmap_camera.todict(),
    }
    camera_info["colmap_camera"]["model"] = camera_info["colmap_camera"]["model"].name
    camera_info["colmap_camera"]["has_prior_focal_length"] = int(
        camera_info["colmap_camera"]["has_prior_focal_length"]
    )
    xml_path = out_dir / "calibration.xml"
    storage = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_WRITE)
    for key, value in camera_info.items():
        if isinstance(value, dict):
            storage.startWriteStruct(key, cv2.FileNode_MAP)
            for subkey, subvalue in value.items():
                storage.write(name=subkey, val=subvalue)
            storage.endWriteStruct()
        else:
            storage.write(name=key, val=value)
    storage.release()

    def gaze_valid(gaze) -> bool:
        combined_valid = getattr(gaze, "combined_gaze_valid", True)
        return bool(combined_valid) and np.isfinite(float(gaze.yaw)) and np.isfinite(float(gaze.pitch))

    def gaze_point_cpf(gaze) -> np.ndarray:
        if getattr(gaze, "spatial_gaze_point_valid", False):
            return as_xyz(gaze.spatial_gaze_point_in_cpf)
        depth = float(getattr(gaze, "depth", 0.0))
        if not np.isfinite(depth) or depth <= 0.0:
            depth = 1.0
        return as_xyz(mps.get_eyegaze_point_at_depth(float(gaze.yaw), float(gaze.pitch), depth))

    def point_in_upright_rgb(point_cpf: np.ndarray) -> np.ndarray:
        point_rgb = as_xyz(T_rgb_cpf @ point_cpf)
        projected = rgb_calib_for_video.project(point_rgb)
        if projected is None:
            return np.full(2, np.nan, dtype=np.float64)
        projected = np.asarray(projected, dtype=np.float64).reshape(-1)
        if projected.size != 2 or not np.all(np.isfinite(projected)):
            return np.full(2, np.nan, dtype=np.float64)
        return projected

    samples: list[np.ndarray] = []
    for gaze in selected_gaze:
        if not gaze_valid(gaze):
            continue

        point_cpf = gaze_point_cpf(gaze)
        pixel_xy = point_in_upright_rgb(point_cpf)

        # Gen 2 provides independent yaw/pitch and validity for each eye.
        vergence = gaze.vergence
        left_origin = np.array([vergence.tx_left_eye, vergence.ty_left_eye, vergence.tz_left_eye])
        right_origin = np.array([vergence.tx_right_eye, vergence.ty_right_eye, vergence.tz_right_eye])
        left_valid = bool(getattr(vergence, "left_gaze_valid", True))
        right_valid = bool(getattr(vergence, "right_gaze_valid", True))

        if getattr(gaze, "spatial_gaze_point_valid", False):
            left_dir = point_cpf - left_origin
            right_dir = point_cpf - right_origin
            left_norm, right_norm = np.linalg.norm(left_dir), np.linalg.norm(right_dir)
            left_dir = left_dir / left_norm if left_norm > 0 else np.full(3, np.nan)
            right_dir = right_dir / right_norm if right_norm > 0 else np.full(3, np.nan)
        else:
            left_dir = unit_gaze_direction(vergence.left_yaw, vergence.left_pitch)
            right_dir = unit_gaze_direction(vergence.right_yaw, vergence.right_pitch)

        if not left_valid:
            left_dir = np.full(3, np.nan)
        if not right_valid:
            right_dir = np.full(3, np.nan)

        left_origin_mm = left_origin * 1000.0
        right_origin_mm = right_origin * 1000.0
        timestamp_us = gaze.tracking_timestamp / datetime.timedelta(microseconds=1)
        relative_timestamp_us = timestamp_us - rgb_start_ns / 1000.0

        samples.append(
            np.concatenate(
                (
                    [relative_timestamp_us],
                    pixel_xy,
                    point_cpf * 1000.0,
                    left_dir,
                    right_dir,
                    left_origin_mm,
                    right_origin_mm,
                )
            )
        )

    columns = [
        "timestamp", "gaze_pos_vid_x", "gaze_pos_vid_y",
        "gaze_pos_3d_x", "gaze_pos_3d_y", "gaze_pos_3d_z",
        "gaze_dir_left_x", "gaze_dir_left_y", "gaze_dir_left_z",
        "gaze_dir_right_x", "gaze_dir_right_y", "gaze_dir_right_z",
        "gaze_ori_left_x", "gaze_ori_left_y", "gaze_ori_left_z",
        "gaze_ori_right_x", "gaze_ori_right_y", "gaze_ori_right_z",
    ]
    gaze_df = pd.DataFrame(samples, columns=columns)
    gaze_df.to_csv(out_dir / "gaze.tsv", sep="\t", float_format="%.8f", index=False, na_rep="nan")

    metadata = provider.get_metadata()
    metadata_out = {
        "device": "Meta Aria Gen2",
        "start_time": metadata.start_time_epoch_sec,
        "glasses_serial": metadata.device_serial,
        "duration": int((max(rgb_end_ns, et_end_ns) - min(rgb_start_ns, et_start_ns)) / 1e6),
        "scene_camera_serial": rgb_config.sensor_serial,
        "name": vrs_path.stem,
        "gaze_source": gaze_source,
    }
    if gaze_source_path is not None:
        metadata_out["gaze_source_file"] = str(gaze_source_path)
    sidecar = Path(str(vrs_path) + ".json")
    if sidecar.is_file():
        with sidecar.open("r", encoding="utf-8") as f:
            sidecar_data = json.load(f)
        if sidecar_data.get("firmware_version"):
            metadata_out["firmware_version"] = sidecar_data["firmware_version"]
        companion_version = sidecar_data.get("companion_version")
        if companion_version:
            metadata_out["recording_software_version"] = companion_version

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata_out, f, indent=2)

    print(f"Done. Exported files to: {out_dir}")
    print(f"  video: {video_path}")
    print(f"  gaze rows: {len(gaze_df)}")


if __name__ == "__main__":
    main()
