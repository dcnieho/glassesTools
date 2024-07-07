import pathlib
import cv2

from . import drawing, gaze_headref, gaze_worldref, intervals, ocv, plane, timestamps, video_gui


def show_visualization(
        working_dir: str|pathlib.Path,
        in_video: str|pathlib.Path, frame_timestamp_file: str|pathlib.Path, camera_calibration_file: str|pathlib.Path,
        planes: dict[str, plane.Plane], poses: dict[str, dict[int, plane.Pose]],
        head_gazes: dict[int, list[gaze_headref.Gaze]], plane_gazes: dict[int, list[gaze_worldref.Gaze]],
        interval_dict: dict[str, list[list[int]]],
        gui: video_gui.GUI, frame_win_id: int, show_planes: bool, show_only_intervals: bool, sub_pixel_fac: int
    ):
    working_dir             = pathlib.Path(working_dir)
    in_video                = pathlib.Path(in_video)
    frame_timestamp_file    = pathlib.Path(frame_timestamp_file)
    camera_calibration_file = pathlib.Path(camera_calibration_file)

    # prep visualizations
    # open video
    cap             = ocv.CV2VideoReader(in_video, timestamps.VideoTimestamps(frame_timestamp_file).timestamps)
    width           = cap.get_prop(cv2.CAP_PROP_FRAME_WIDTH)
    height          = cap.get_prop(cv2.CAP_PROP_FRAME_HEIGHT)
    gui.set_framerate(cap.get_prop(cv2.CAP_PROP_FPS))
    cam_params      = ocv.CameraParams.readFromFile(camera_calibration_file)

    # add windows for planes, if wanted
    if show_planes:
        plane_win_id = {p: gui.add_window(p) for p in planes}

    stopAllProcessing = False
    max_frame_idx = max(head_gazes.keys())
    for frame_idx in range(max_frame_idx+1):
        done, frame, frame_idx, frame_ts = cap.read_frame(report_gap=True)
        if done or intervals.beyond_last_interval(frame_idx, interval_dict):
            break

        keys = gui.get_key_presses()
        if 'q' in keys:
            # quit fully
            stopAllProcessing = True
            break
        if 'n' in keys:
            # goto next
            break

        # check we're in a current interval, else skip processing
        # NB: have to spool through like this, setting specific frame to read
        # with cap.get(cv2.CAP_PROP_POS_FRAMES) doesn't seem to work reliably
        # for VFR video files
        if show_only_intervals and not intervals.is_in_interval(frame_idx, interval_dict):
            # no need to show this frame
            continue

        if show_planes:
            refImg = {p: planes[p].get_ref_image(400) for p in planes}

        if frame_idx in head_gazes:
            for gaze_head in head_gazes[frame_idx]:
                # draw gaze point on scene video
                gaze_head.draw(frame, cam_params, sub_pixel_fac)

                # draw plane gazes on video and plane
                for p in planes:
                    if frame_idx in plane_gazes[p]:
                        for gaze_world in plane_gazes[p][frame_idx]:
                            gaze_world.drawOnWorldVideo(frame, cam_params, sub_pixel_fac)
                            if show_planes:
                                gaze_world.drawOnPlane(refImg[p], planes[p], sub_pixel_fac)

        if show_planes:
            for p in planes:
                gui.update_image(refImg[p], frame_ts/1000., frame_idx, window_id = plane_win_id[p])

        # if we have poster pose, draw poster origin on video
        for p in planes:
            if frame_idx in poses[p]:
                a = poses[p][frame_idx].getOriginOnImage(cam_params)
                drawing.openCVCircle(frame, a, 3, (0,255,0), -1, sub_pixel_fac)
                drawing.openCVLine(frame, (a[0],0), (a[0],height), (0,255,0), 1, sub_pixel_fac)
                drawing.openCVLine(frame, (0,a[1]), (width,a[1]) , (0,255,0), 1, sub_pixel_fac)

        # keys is populated above
        if 's' in keys:
            # screenshot
            cv2.imwrite(working_dir / f'project_frame_{frame_idx}.png', frame)

        gui.update_image(frame, frame_ts/1000., frame_idx, window_id = frame_win_id)
        closed, = gui.get_state()
        if closed:
            stopAllProcessing = True
            break

    gui.stop()

    return stopAllProcessing