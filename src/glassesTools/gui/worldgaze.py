import pathlib

from .. import annotation, drawing, gaze_headref, gaze_worldref, intervals, ocv, plane, timestamps
from . import video_player


def show_visualization(
        in_video: str|pathlib.Path, frame_timestamp_file: str|pathlib.Path, camera_calibration_file: str|pathlib.Path,
        planes: dict[str, plane.Plane], poses: dict[str, dict[int, plane.Pose]],
        head_gazes: dict[int, list[gaze_headref.Gaze]], plane_gazes: dict[str, dict[int, list[gaze_worldref.Gaze]]],
        annotations: dict[annotation.Event, list[list[int]]],
        gui: video_player.GUI, show_planes: bool, show_only_intervals: bool, sub_pixel_fac: int
    ):
    in_video                = pathlib.Path(in_video)
    frame_timestamp_file    = pathlib.Path(frame_timestamp_file)
    camera_calibration_file = pathlib.Path(camera_calibration_file)

    # prep visualizations
    # open video
    video_ts        = timestamps.VideoTimestamps(frame_timestamp_file)
    cap             = ocv.CV2VideoReader(in_video, video_ts.timestamps)
    cam_params      = ocv.CameraParams.read_from_file(camera_calibration_file)

    # flatten if needed
    annotations_flat: dict[annotation.Event, list[int]] = {}
    for e in annotations:
        if annotations[e] and isinstance(annotations[e][0],list):
            annotations_flat[e] = [i for iv in annotations[e] for i in iv]
        else:
            annotations_flat[e] = annotations[e].copy()
    gui.set_show_timeline(True, video_ts, annotations_flat, window_id=gui.main_window_id)

    # add windows for planes, if wanted
    if show_planes:
        plane_win_id = {p: gui.add_window(p) for p in planes}

    max_frame_idx = max(head_gazes.keys())
    should_exit = False
    first_frame = True
    for frame_idx in range(max_frame_idx+1):
        done, frame, frame_idx, frame_ts = cap.read_frame(report_gap=True)
        if first_frame and frame is not None:
            gui.set_frame_size(frame.shape, gui.main_window_id)
            first_frame = False
        if done or intervals.beyond_last_interval(frame_idx, annotations):
            break

        requests = gui.get_requests()
        for r,p in requests:
            if r=='exit':   # only request we need to handle
                should_exit = True
                break
        if should_exit:
            break

        # check we're in a current interval, else skip processing
        # NB: have to spool through like this, setting specific frame to read
        # with cap.get(cv2.CAP_PROP_POS_FRAMES) doesn't seem to work reliably
        # for VFR video files
        if show_only_intervals and not intervals.is_in_interval(frame_idx, annotations) or frame is None:
            # we don't have a valid frame or no need to show this frame
            # do update timeline of the viewers
            gui.update_image(None, frame_ts/1000., frame_idx, window_id=gui.main_window_id)
            if show_planes:
                for p in planes:
                    gui.update_image(None, frame_ts/1000., frame_idx, window_id=plane_win_id[p])
            continue

        if show_planes:
            ref_img = {p: planes[p].get_ref_image(400) for p in planes}

        if frame_idx in head_gazes:
            for gaze_head in head_gazes[frame_idx]:
                # draw gaze point on scene video
                gaze_head.draw(frame, cam_params, sub_pixel_fac)

        # draw plane gazes on video and plane
        for p in planes:
            if frame_idx in plane_gazes[p]:
                for gaze_world in plane_gazes[p][frame_idx]:
                    gaze_world.draw_on_world_video(frame, cam_params, sub_pixel_fac)
                    if show_planes:
                        gaze_world.draw_on_plane(ref_img[p], planes[p], sub_pixel_fac)

        if show_planes:
            for p in planes:
                gui.update_image(ref_img[p], frame_ts/1000., frame_idx, window_id=plane_win_id[p])

        # if we have plane pose, draw plane origin on video
        for p in planes:
            if frame_idx in poses[p]:
                a = poses[p][frame_idx].get_origin_on_image(cam_params)
                drawing.openCVCircle(frame, a, 3, (0,255,0), -1, sub_pixel_fac)
                ll = 20
                drawing.openCVLine(frame, (a[0],a[1]-ll), (a[0],a[1]+ll), (0,255,0), 1, sub_pixel_fac)
                drawing.openCVLine(frame, (a[0]-ll,a[1]), (a[0]+ll,a[1]), (0,255,0), 1, sub_pixel_fac)

        gui.update_image(frame, frame_ts/1000., frame_idx, window_id=gui.main_window_id)

    gui.stop()